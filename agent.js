import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import proc from "node:process";
import * as readline from "node:readline/promises";

const model = new ChatOpenAI({
  temperature: 0.7,
});

// const model = new ChatFireworks({
//   temperature: 0.7,
// });

const prompt = ChatPromptTemplate.fromMessages([
  ("system", "You are a helpful assistant called Max."),
  new MessagesPlaceholder("chat_history"),
  ("human", "{input}"),
  new MessagesPlaceholder("agent_scratchpad"),
]);

const searchTool = new TavilySearchResults();
const tools = [searchTool];

model.bindTools(tools);
const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools,
});

const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

const chatHistory = [];

const readInput = async () => {
  // Read user input
  const rl = readline.createInterface({
    input: proc.stdin,
    output: proc.stdout,
  });
  const userPrompt = await rl.question("@alcatraz627:");
  if (userPrompt.trim() === "") {
    return;
  }

  const resp = await agentExecutor.invoke({
    input: userPrompt || "Tell me a joke",
    chat_history: chatHistory,
  });
  console.log("Animus Silica: " + resp.output);
  chatHistory.push(new HumanMessage(userPrompt || "Tell me a joke"));
  chatHistory.push(new AIMessage(resp.output));
  readInput();
};

readInput();
