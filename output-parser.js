import { ChatFireworks } from "@langchain/community/chat_models/fireworks";
import {
  CommaSeparatedListOutputParser,
  StringOutputParser,
  StructuredOutputParser,
} from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import proc from "node:process";
import * as readline from "node:readline/promises";

const model = new ChatFireworks({
  // Use this model for list and json
  // model: "accounts/fireworks/models/mixtral-8x7b-instruct",
  temperature: 0.7,
});

const rl = readline.createInterface({
  input: proc.stdin,
  output: proc.stdout,
});

const callStringOutputParser = async () => {
  const userPrompt = await rl.question("O' mighty statistical sentience, ");

  const prompt = ChatPromptTemplate.fromMessages([
    //   "You are a comedian who keeps the responses short. Include the word {input}"
    [
      "system",
      "Generate a quote based on the word provided by the user, and cite the author",
    ],
    ["human", "{input}"],
  ]);
  // console.log(await prompt.format({ input: userPrompt }));

  const parser = new StringOutputParser();

  const chain = prompt.pipe(model).pipe(parser);

  const resp = await chain.invoke({ input: userPrompt });

  return resp;
};

const callListOutputParser = async () => {
  const prompt = ChatPromptTemplate.fromTemplate(
    `Provide 5 similar words, separated by commas, for the word {word}. Skip the confirmation message of the task.`
  );
  const outputParser = new CommaSeparatedListOutputParser();

  const userPrompt = await rl.question("Quote Generator ");

  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({ word: userPrompt });
};

const callStructuredOutputParser = async () => {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract information from the following phrase.
    Formatting Instructions: {format_instr}
    Phrase: {phrase}
  `);
  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "Name of the person.",
    age: "Age of the person. Prefer a number.",
    profession: "Profession of the person. Infer from action",
  });

  const userPrompt = await rl.question("Phrase ");
  const chain = prompt.pipe(model).pipe(outputParser);

  return await chain.invoke({
    phrase: userPrompt,
    format_instr: outputParser.getFormatInstructions(),
  });
};

// const resp = callStringOutputParser();
// const resp = await callListOutputParser();
const resp = await callStructuredOutputParser();
console.log(resp);

rl.close();
