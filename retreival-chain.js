import { ChatFireworks } from "@langchain/community/chat_models/fireworks";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { FireworksEmbeddings } from "@langchain/community/embeddings/fireworks";
import { AIMessageChunk, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import proc from "node:process";
import * as readline from "node:readline/promises";

// const documentA = new Document({
//   pageContent: ctx,
// });

const createVectorStore = async () => {
  const loader = new CheerioWebBaseLoader(
    // "https://en.wikipedia.org/wiki/2024_Indian_general_election",
    "https://en.wikipedia.org/wiki/Holland"
    // "https://readme.fireworks.ai/docs/structured-response-formatting/"
  );

  const docs = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 50,
  });
  const splitDocs = await splitter.splitDocuments(docs);
  const embeddings = new FireworksEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  return vectorStore;
};

const chatHistory = [
  new HumanMessage("Answer my question like a pirate. " + ""),
  new AIMessageChunk("Aye aye captain"),
];

const createChain = async (vectorStore) => {
  const model = new ChatFireworks({
    temperature: 0.7,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    ["system", "Answer the user's question based on the context: {context}"],
  ]);

  // const chain = prompt.pipe(model);
  const chain = await createStuffDocumentsChain({ llm: model, prompt });

  const retriever = vectorStore.asRetriever({
    k: 3,
  });

  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    // ["user", ""],
    [
      "user",
      `Given the above conversation, generate a search query to look up in order to get the answer
      Input: {input}`,
    ],
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrieverPrompt,
  });

  // Prepare final chain
  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever,
  });

  return conversationChain;
};

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

// Read user input
const rl = readline.createInterface({
  input: proc.stdin,
  output: proc.stdout,
});
const userPrompt = await rl.question("O' mighty statistical sentience, ");

// Chat history

// Invoke the deity
const resp = await chain.invoke({
  chat_history: chatHistory,
  input: userPrompt || "Tell me about the history of holland.",
});
console.log(resp.answer);

rl.close();
