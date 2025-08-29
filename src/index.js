import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { exec } from 'child_process';
import Database from 'better-sqlite3';

import { ChatOpenAI } from '@langchain/openai';
import { StateGraph, START, END } from '@langchain/langgraph';

const modelName = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const llm = new ChatOpenAI({ model: modelName, temperature: 0.2 });

const ROOT = process.cwd();
const DOCS_DIR = path.join(ROOT, 'data', 'documents');
const DB_DIR = path.join(ROOT, 'data', 'sqlite');

// ---------- util: CLI ----------
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const ask = (q) => new Promise((res) => rl.question(q, (ans) => res(ans.trim())));

function loadTxtDocs() {
  if (!fs.existsSync(DOCS_DIR)) return [];
  const files = fs.readdirSync(DOCS_DIR).filter((f) => f.endsWith('.txt'));
  return files.map((file) => ({
    file,
    text: fs.readFileSync(path.join(DOCS_DIR, file), 'utf-8')
  }));
}

function searchDocs(question, docs, maxChars = 1800) {
  // Busca simples por palavras da pergunta; junta trechos relevantes.
  const qTerms = question.toLowerCase().split(/\W+/).filter(Boolean);
  const scored = [];
  for (const d of docs) {
    const lines = d.text.split(/\r?\n/);
    for (const line of lines) {
      const L = line.toLowerCase();
      const score = qTerms.reduce((acc, t) => acc + (L.includes(t) ? 1 : 0), 0);
      if (score > 0) scored.push({ file: d.file, score, line });
    }
  }
  scored.sort((a, b) => b.score - a.score);
  const snippet = scored.slice(0, 20).map(s => `(${s.file}) ${s.line}`).join('\n');
  return snippet.slice(0, maxChars);
}

// ---------- sqlite ----------
function findDbFiles() {
  if (!fs.existsSync(DB_DIR)) return [];
  return fs.readdirSync(DB_DIR).filter((f) => f.endsWith('.db')).map(f => path.join(DB_DIR, f));
}

function sqliteSchema(dbPath) {
  const db = new Database(dbPath, { readonly: true });
  const rows = db.prepare(`SELECT name, sql FROM sqlite_master WHERE type='table'`).all();
  db.close();
  return rows.map(r => `-- ${r.name}\n${r.sql}`).join('\n\n');
}

async function proposeSql(question, schema) {
  const prompt = [
    {
      role: 'system',
      content:
        `Você é um assistente que gera uma única query SQL (somente SELECT) para SQLite, ` +
        `com base no esquema fornecido. Regras:\n` +
        `- Use apenas SELECT (sem INSERT/UPDATE/DELETE)\n` +
        `- Limite o resultado com LIMIT 20\n` +
        `- Retorne APENAS a query crua, sem explicações nem markdown.\n`
    },
    {
      role: 'user',
      content: `Pergunta: ${question}\n\nEsquema:\n${schema}`
    }
  ];
  const out = await llm.invoke(prompt);
  return String(out.content).trim();
}

function runSql(dbPath, sql) {
  const db = new Database(dbPath, { readonly: true });
  try {
    const rows = db.prepare(sql).all();
    return { ok: true, rows };
  } catch (err) {
    return { ok: false, error: String(err) };
  } finally {
    db.close();
  }
}

// ---------- bash (curl) ----------
async function proposeCurl(question) {
  const prompt = [
    {
      role: 'system',
      content:
        `Proponha UM comando bash seguro baseado em curl para obter informação relevante ` +
        `à pergunta do usuário. Restrições:\n` +
        `- Deve começar com: curl -s \n` +
        `- Pode incluir -L, -H e URL; sem pipes, sem redirecionamentos, sem subshell.\n` +
        `- Responda APENAS com o comando, sem explicações.`
    },
    { role: 'user', content: question }
  ];
  const out = await llm.invoke(prompt);
  return String(out.content).trim();
}

function isSafeCurl(cmd) {
  return /^curl\s+-s(\s+-L|\s+-H\s+'[^']+'|\s+-H\s+"[^"]+"|\s+--header\s+\"[^\"]+\")*\s+https?:\/\/\S+$/i.test(cmd);
}

async function runBashWithApproval(cmd) {
  console.log(`\nPlano (bash): ${cmd}`);
  const ok = (await ask('Autorizar execução? [y/N] ')).toLowerCase() === 'y';
  if (!ok) return { approved: false, output: '' };

  return await new Promise((resolve) => {
    exec(cmd, { maxBuffer: 1024 * 1024 }, (err, stdout, stderr) => {
      if (err) resolve({ approved: true, output: `ERRO: ${String(err)}` });
      else resolve({ approved: true, output: (stdout || stderr || '').slice(0, 4000) });
    });
  });
}

// ---------- roteador ----------
function heuristicRoute(question) {
  const q = question.toLowerCase();
  const docHints = ['econom', 'book', 'livro', 'smith', 'keynes', 'marx', 'hayek', 'piketty'];
  const dbHints = ['artista','artistas','faixa','faixas','gênero','genero','álbum','album','álbuns','albuns', 'faturamento','venda', 'preço','preços','preco','precos','médio','media','média','valor','custo', 'music','song','track','artist','album','playlist','genre',];
  const webHints = ['http', 'site', 'web', 'url', 'baixar', 'fetch', 'curl'];

  const score = (arr) => arr.reduce((a, k) => a + (q.includes(k) ? 1 : 0), 0);
  const sDoc = score(docHints);
  const sDb  = score(dbHints);
  const sWeb = score(webHints);

  const top = Math.max(sDoc, sDb, sWeb);
  const picks = [];
  if (sDoc === top && top > 0) picks.push('documents');
  if (sDb  === top && top > 0) picks.push('sqlite');
  if (sWeb === top && top > 0) picks.push('bash');

  if (picks.length === 0) return 'documents';      // fallback: docs
  if (picks.length > 1)  return 'combine';
  return picks[0];
}

/** @typedef {{ question: string, route?: string, context?: string[], final?: string, steps?: string[] }} AgentState */

const graph = new StateGraph({
  channels: {
    question: { value: (_prev, next) => next, default: () => '' },
    route:    { value: (_prev, next) => next, default: () => '' },
    context:  { value: (prev, next) => (prev ?? []).concat(next ?? []), default: () => [] },
    final:    { value: (_prev, next) => next, default: () => '' },
    steps:    { value: (prev, next) => (prev ?? []).concat(next ?? []), default: () => [] }
  }
});


graph.addNode('router', async (state) => {
  const route = heuristicRoute(state.question);
  return { route, steps: [`route=${route}`] };
});


graph.addNode('sqlite', async (state) => {
  const dbFiles = findDbFiles();
  if (dbFiles.length === 0) return { context: [`[sqlite] Nenhum .db encontrado em ${DB_DIR}`] };

  // heurística simples: usa o primeiro .db
  const dbPath = dbFiles[0];
  const schema = sqliteSchema(dbPath);
  const sql = await proposeSql(state.question, schema);
  const safe = /^\s*select/i.test(sql);
  if (!safe) return { context: [`[sqlite] Query rejeitada: ${sql}`], steps: [`sql(rejected)`] };

  const res = runSql(dbPath, sql);
  if (!res.ok) return { context: [`[sqlite] ERRO: ${res.error}`], steps: [`sql(error)`] };

  const preview = JSON.stringify(res.rows.slice(0, 20), null, 2);
  return { context: [`[sqlite] DB=${path.basename(dbPath)}\nSQL: ${sql}\nRESULT:\n${preview}`], steps: [`sql(ok)`] };
});

graph.addNode('documents', async (state) => {
  const docs = loadTxtDocs();
  if (docs.length === 0) return { context: [`[docs] Nenhum .txt encontrado em ${DOCS_DIR}`] };
  const snippet = searchDocs(state.question, docs);
  return { context: [`[docs]\n${snippet}`], steps: ['docs(ok)'] };
});

graph.addNode('bash', async (state) => {
  const cmd = await proposeCurl(state.question);
  if (!isSafeCurl(cmd)) {
    return { context: [`[bash] Comando não seguro ou inválido: ${cmd}`], steps: ['bash(rejected)'] };
  }
  const { approved, output } = await runBashWithApproval(cmd);
  if (!approved) return { context: ['[bash] Execução cancelada pelo usuário.'], steps: ['bash(cancel)'] };
  return { context: [`[bash]\nCMD: ${cmd}\nOUTPUT:\n${output}`], steps: ['bash(ok)'] };
});


graph.addNode('combine', async (state) => {
  const a = await graph.getNode('sqlite').func(state);
  const b = await graph.getNode('documents').func(state);
  return { context: [...(a.context||[]), ...(b.context||[])], steps: ['combine'] };
});

// resposta final
graph.addNode('answer', async (state) => {
  const ctx = (state.context || []).join('\n\n');
  const prompt = [
    { role: 'system', content: 'Responda de forma clara, citando de qual fonte veio cada parte (sqlite/docs/bash) quando possível.' },
    { role: 'user', content: `Pergunta: ${state.question}\n\nContexto disponível:\n${ctx}\n\nResponda objetivamente.` }
  ];
  const out = await llm.invoke(prompt);
  return { final: String(out.content).trim() };
});

// liga o grafo
graph.addEdge(START, 'router');
graph.addConditionalEdges('router', (s) => s.route, {
  sqlite: 'sqlite',
  documents: 'documents',
  bash: 'bash',
  combine: 'combine'
});
graph.addEdge('sqlite', 'answer');
graph.addEdge('documents', 'answer');
graph.addEdge('bash', 'answer');
graph.addEdge('combine', 'answer');
graph.addEdge('answer', END);

const app = graph.compile();

// ---------- CLI loop ----------
(async () => {
  console.log(`Multi-Source Agent (${modelName})`);
  console.log(`Docs: ${DOCS_DIR}`);
  console.log(`DBs : ${DB_DIR}`);
  console.log(`Digite sua pergunta (ou "sair")\n`);
  while (true) {
    const q = await ask('Você: ');
    if (!q || q.toLowerCase() === 'sair') break;
    const res = await app.invoke({ question: q });
    console.log('\nAgente:\n' + res.final + '\n');
  }
  rl.close();
})();
