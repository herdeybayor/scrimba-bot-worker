import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { createClient } from '@supabase/supabase-js';

import type { Document } from '@langchain/core/documents';

function combineDocuments(docs: Document[]): string {
	return docs.map((doc) => doc.pageContent).join('\n\n');
}

function formatConvHistory(messages: string[]): string {
	return messages.map((message, i) => (i % 2 === 0 ? `Human: ${message}` : `AI: ${message}`)).join('\n');
}

const corsHeaders = {
	'Access-Control-Allow-Origin': '*',
	'Access-Control-Allow-Methods': 'POST, OPTIONS',
	'Access-Control-Allow-Headers': 'Content-Type',
};

export default {
	async fetch(request, env, ctx): Promise<Response> {
		// Handle CORS preflight request
		if (request.method === 'OPTIONS') {
			return new Response(null, {
				headers: corsHeaders,
			});
		}

		try {
			const { question, conv_history } = await request.json<Record<string, any>>();

			if (!question) {
				return new Response('Missing question', { status: 400, headers: corsHeaders });
			}

			if (!conv_history) {
				return new Response('Missing conversation history', { status: 400, headers: corsHeaders });
			}

			const openAIApiKey = env.OPENAI_API_KEY;
			const openAIBaseURL = env.OPENAI_BASE_URL;
			const sbApiKey = env.SUPABASE_API_KEY;
			const sbUrl = env.SUPABASE_URL;

			const client = createClient(sbUrl, sbApiKey);
			const embeddings = new OpenAIEmbeddings({ openAIApiKey });

			const vectorStore = new SupabaseVectorStore(embeddings, {
				client,
				tableName: 'documents',
				queryName: 'match_documents',
			});

			const retriever = vectorStore.asRetriever();
			const llm = new ChatOpenAI({ openAIApiKey, model: 'gpt-4o-mini', configuration: { baseURL: openAIBaseURL } });

			const standaloneQuestionTemplate = `Given some conversation history (if any) and a question, convert the question to a standalone question. 
            conversation history: {conv_history}
            question: {question} 
            standalone question:`;
			const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate);

			const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided and the conversation history. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
            context: {context}
            conversation history: {conv_history}
            question: {question}
            answer: `;
			const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

			const standaloneQuestionChain = standaloneQuestionPrompt.pipe(llm).pipe(new StringOutputParser());

			const retrieverChain = RunnableSequence.from([(prevResult) => prevResult.standalone_question, retriever, combineDocuments]);
			const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

			const chain = RunnableSequence.from([
				{
					standalone_question: standaloneQuestionChain,
					original_input: new RunnablePassthrough(),
				},
				{
					context: retrieverChain,
					question: ({ original_input }) => original_input.question,
					conv_history: ({ original_input }) => original_input.conv_history,
				},
				answerChain,
			]);

			const result = await chain.invoke({ question, conv_history: formatConvHistory(conv_history) });

			return new Response(
				JSON.stringify({
					question,
					answer: result,
				}),
				{
					headers: {
						'Content-Type': 'application/json',
						...corsHeaders,
					},
				}
			);
		} catch (error) {
			console.error(error);
			return new Response('Error processing request', {
				status: 500,
				headers: corsHeaders,
			});
		}
	},
} satisfies ExportedHandler<Env>;
