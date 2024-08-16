from langchain_cohere import ChatCohere


class LLMChain:
    def get_response(self, input_prompt):
        pass


class CohereLLMChain(LLMChain):
    def __init__(self):
        self.llm_chain = ChatCohere()

    def get_llm(self):
        return self.llm_chain

    def get_response(self, input_prompt):
        return self.get_llm().invoke({"input": input_prompt})
