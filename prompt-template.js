import { ChatFireworks } from "@langchain/community/chat_models/fireworks";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import proc from "node:process";
import * as readline from "node:readline/promises";

const model = new ChatFireworks({
  temperature: 0.3,
});

const rl = readline.createInterface({ input: proc.stdin, output: proc.stdout });
const userPrompt = await rl.question("O' mighty statistical sentience, ");

const prompt = ChatPromptTemplate.fromMessages([
  //   "You are a comedian who keeps the responses short. Include the word {input}"
  ["system", "Generate a quote based on the word provided by the user"],
  ["human", "{input}"],
]);

// console.log(await prompt.format({ input: userPrompt }));

const chain = prompt.pipe(model);

const resp = await chain.invoke({ input: userPrompt });

console.log(resp);

rl.close();
