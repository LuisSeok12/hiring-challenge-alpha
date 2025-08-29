# Multi-Source AI Agent

Agente conversacional em Node.js que responde perguntas combinando **SQLite**, **documentos `.txt`** e **web (via bash `curl`)** com aprovação do usuário. Orquestrado com **LangGraph**, LLM via **LangChain**.

## Requisitos

* Node 18+ (testado em Node 20)
* Uma chave da OpenAI em `OPENAI_API_KEY`

## Instalação

```bash
git clone <seu-fork>
cd hiring-challenge-alpha
cp .env.example .env       # preencha OPENAI_API_KEY
npm install
```

## Estrutura

```
data/
  documents/
    economy_books.txt
  sqlite/
    music.db
src/
  index.js
.env (.env.example)
package.json
```

## Executar

```bash
npm start
```

Você verá:

```
Multi-Source Agent (gpt-4o-mini)
Docs: .../data/documents
DBs : .../data/sqlite
Digite sua pergunta (ou "sair")
```

## Como funciona

* **Roteamento**: uma heurística decide entre `sqlite`, `documents`, `bash` ou `combine`.
* **SQLite**: 

  * *Preço médio das faixas por gênero*
  * *Quantidade de músicas em álbuns “Greatest Hits”*
* **Documentos**: busca trechos relevantes em `.txt`.
* **Bash**: propõe `curl -s <URL>` e **pede sua aprovação** antes de executar.

## Testes sugeridos

**SQLite**

* “Quais são os 5 artistas com mais faixas?”
* “Preço médio das faixas por gênero”
* “Quantas músicas tem o álbum Greatest Hits?”

**Documentos**

* “Quem escreveu ‘A Riqueza das Nações’?”
* “Explique a tese de Piketty sobre desigualdade.”

**Bash (com aprovação)**

* “Baixe o HTML de [https://example.com](https://example.com) e resuma o título.”

**Combine (múltiplas fontes)**

* “Resuma a tese de Piketty e mostre também os 3 artistas com mais faixas.”

> Dica: termos como *artista, faixa, gênero, álbum* puxam SQLite; *book/livro/nome de economistas* puxam documentos; *http, web, curl* puxam bash.


