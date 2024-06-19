import { ChatFireworks } from "@langchain/community/chat_models/fireworks";
import proc from "node:process";
import * as readline from "node:readline/promises";

const model = new ChatFireworks({
  apiKey: process.env.FIREWORKS_API_KEY,
  temperature: 0,
});

const rl = readline.createInterface({ input: proc.stdin, output: proc.stdout });
const userPrompt = await rl.question("O' mighty statistical sentience, ");

const res = await model.stream(
  userPrompt || "What planet are we on? Write a poem on it"
);

rl.close();

// console.log(res);
for await (const chunk of res) {
  console.log(chunk.content);
}
