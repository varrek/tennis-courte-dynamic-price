from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class TextProcessor:
    def __init__(self, api_key):
        self.llm = OpenAI(api_key=api_key)
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract tennis court booking details from the following text. 
            Return a JSON with the following fields (if mentioned):
            - duration (in hours)
            - court_surface (Hard/Clay/Grass/Carpet)
            - court_type (Indoor/Outdoor)
            - num_players (2 or 4)
            - match_type (Singles/Doubles/Training)
            - coaching_requested (true/false)
            - ball_machine (true/false)
            - court_quality (Standard/Premium/Elite)
            
            Text: {text}
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def process_text(self, text):
        try:
            result = self.chain.run(text)
            return result
        except Exception as e:
            return f"Error processing text: {str(e)}" 