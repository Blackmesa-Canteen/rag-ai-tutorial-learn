import { WorkflowEntrypoint } from "cloudflare:workers";
import { Hono } from "hono";
import Anthropic from '@anthropic-ai/sdk';
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";


const app = new Hono();

app.get('/', async (c) => {
	// ... Existing code
	const systemPrompt = `When answering the question or responding, use the context provided, if it is provided and relevant.`

	let modelUsed = ""
	let response = null

	if (c.env.ANTHROPIC_API_KEY) {
		const anthropic = new Anthropic({
			apiKey: c.env.ANTHROPIC_API_KEY
		})

		const model = "claude-3-5-sonnet-latest"
		modelUsed = model

		const message = await anthropic.messages.create({
			max_tokens: 1024,
			model,
			messages: [
				{ role: 'user', content: question }
			],
			system: [systemPrompt, notes ? contextMessage : ''].join(" ")
		})

		response = {
			response: message.content.map(content => content.text).join("\n")
		}
	} else {
		const model = "@cf/meta/llama-3.1-8b-instruct"
		modelUsed = model

		response = await c.env.AI.run(
			model,
			{
				messages: [
					...(notes.length ? [{ role: 'system', content: contextMessage }] : []),
					{ role: 'system', content: systemPrompt },
					{ role: 'user', content: question }
				]
			}
		)
	}

	if (response) {
		c.header('x-model-used', modelUsed)
		return c.text(response.response)
	} else {
		return c.text("We were unable to generate output", 500)
	}
})

app.post('/notes', async (c) => {
	const { text } = await c.req.json();
	if (!text) return c.text("Missing text", 400);
	await c.env.RAG_WORKFLOW.create({ params: { text } })
	return c.text("Created note", 201);
})

app.delete("/notes/:id", async (c) => {
	const { id } = c.req.param();

	const query = `DELETE FROM notes WHERE id = ?`;
	await c.env.DB.prepare(query).bind(id).run();

	await c.env.VECTOR_INDEX.deleteByIds([id]);

	return c.status(204);
});

app.onError((err, c) => {
	return c.text(err);
});


export default app;


export class RAGWorkflow extends WorkflowEntrypoint {
	async run(event, step) {
		const env = this.env
		const { text } = event.payload;
		let texts = await step.do('split text', async () => {
			const splitter = new RecursiveCharacterTextSplitter();
			const output = await splitter.createDocuments([text]);
			return output.map(doc => doc.pageContent);
		})

		console.log("RecursiveCharacterTextSplitter generated ${texts.length} chunks")

		for (const index in texts) {
			const text = texts[index]
			const record = await step.do(`create database record: ${index}/${texts.length}`, async () => {
				const query = "INSERT INTO notes (text) VALUES (?) RETURNING *"

				const { results } = await env.DB.prepare(query)
					.bind(text)
					.run()

				const record = results[0]
				if (!record) throw new Error("Failed to create note")
				return record;
			})

			const embedding = await step.do(`generate embedding: ${index}/${texts.length}`, async () => {
				const embeddings = await env.AI.run('@cf/baai/bge-base-en-v1.5', { text: text })
				const values = embeddings.data[0]
				if (!values) throw new Error("Failed to generate vector embedding")
				return values
			})

			await step.do(`insert vector: ${index}/${texts.length}`, async () => {
				return env.VECTOR_INDEX.upsert([
					{
						id: record.id.toString(),
						values: embedding,
					}
				]);
			})
		}
	}
}