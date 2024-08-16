from langchain_core.prompts import ChatPromptTemplate


class Prompt:
    def get_template(self):
        pass


class GeneratorPrompt(Prompt):
    generator_prompt = """
    Create a cover letter in the Format structure given below. Use the given 2 inputs below as variable to add context to the format.
    Format structure of cover letter:

    Dear Hiring Team,

    Instruction to write Body of letter:
    - Write cover letter body that sounds professional, natural & written by human
    - body of letter in the range of 150 to 250 words. 
    - Body should be written based on steps 1 to 3 mentioned next:
    - Step 1: Extract important key words from input 2 given below
    - Step 2: Extract details about the company from input 2 given below + internet (vision, key products etc.)
    - Step 3: Use extraction from step 1 & step 2 to combine with input 1 given below as context to write cover letter
    - Don’t use terms like “as mentioned in the job description”

    Best Regards

    Input 1 - Resume of applying candidate: {context}
    Input 2 - Job description for the job to apply: {input}
    Note: The output should only contain the cover letter as given in the format structure. 
    Note 2: The last line of the cover letter should contain only the candidate's name Example: "John Doe"
    """

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(self.generator_prompt)

    def get_template(self):
        return self.prompt


class LinkedInMessagePrompt(Prompt):
    generator_prompt = """
        Create a LinkedIn message in the Format structure given below. 
        Use the given 2 inputs below as variable to add context to the format.
        
        Format structure of the message:
        
        Dear Hiring Team,

        Instruction to write Body of letter:
        - Write a message that sounds professional, natural & written by human
        - body of message in the range of 20 to 40 words. 
        - Body should be written based on steps 1 to 3 mentioned next:
        - Step 1: Extract important key words from input 2 given below
        - Step 2: Extract details about the company from input 2 given below + internet (vision, key products etc.)
        - Step 3: Use extraction from step 1 & step 2 to combine with input 1 given below as context to write cover letter
        - Don’t use terms like “as mentioned in the job description”

        Best Regards

        Input 1 - Resume of applying candidate: {context}
        Input 2 - Job description for the job to apply: {input}
        Note: The output should only contain the message as given in the format structure. 
        Note 2: The last line of the message should contain only the candidate's name Example: "John Doe"
        """

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(self.generator_prompt)

    def get_template(self):
        return self.prompt
