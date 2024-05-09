from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from langchain_openai import ChatOpenAI,OpenAI
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

def get_examples():
  examples = [
    # LinkedIn Post format
    {"format": "LinkedIn Post", "topic": "Machine Learning", "output": """Title: Understanding Machine Learning: A Brief Overview

Ever wondered how Netflix suggests your next binge-worthy show or how your email filters out spam? It's all thanks to machine learning (ML). In essence, ML is a branch of artificial intelligence where systems learn from data to make decisions or predictions without being explicitly programmed. It works by collecting and preprocessing data, training models, evaluating their performance, and deploying them for real-world use. There are three main types: supervised, unsupervised, and reinforcement learning. From recommendation systems to medical diagnosis and autonomous vehicles, ML is transforming various industries, making our lives easier and more efficient. #MachineLearning #AI #Technology #machinelearning #futureofwork""" },
    # LinkedIn Post format
    {"format": "LinkedIn Post", "topic": "Generative AI", "output": "Excited to share insights on Generative AI! 🚀 With its innovative capabilities, Generative AI is revolutionising various industries, from creative arts to healthcare. Its potential to generate realistic images, text, and even music is reshaping the way we create and interact with technology. Let's explore the endless possibilities and opportunities this cutting-edge technology offers. #GenerativeAI #Innovation #TechRevolution 🤖💡"},
    # Email format
    {"format": "Email", "topic": "Meeting Reminder - Project Discussion", "output": "Subject: Reminder: Project Discussion Meeting Today at 2 PM. Let's discuss the next steps for [Project Name]. See you there!"},
    # Article format
    {"format": "Article", "topic": "What Is generative Ai?", "output": """Generative AI is a type of artificial intelligence technology that can produce various types of content, including text, imagery, audio and synthetic data. The recent buzz around generative AI has been driven by the simplicity of new user interfaces for creating high-quality text, graphics and videos in a matter of seconds.

  The technology, it should be noted, is not brand-new. Generative AI was introduced in the 1960s in chatbots. But it was not until 2014, with the introduction of generative adversarial networks, or GANs -- a type of machine learning algorithm -- that generative AI could create convincingly authentic images, videos and audio of real people.

  On the one hand, this newfound capability has opened up opportunities that include better movie dubbing and rich educational content. It also unlocked concerns about deepfakes -- digitally forged images or videos -- and harmful cybersecurity attacks on businesses, including nefarious requests that realistically mimic an employee's boss.

  Two additional recent advances that will be discussed in more detail below have played a critical part in generative AI going mainstream: transformers and the breakthrough language models they enabled. Transformers are a type of machine learning that made it possible for researchers to train ever-larger models without having to label all of the data in advance. New models could thus be trained on billions of pages of text, resulting in answers with more depth. In addition, transformers unlocked a new notion called attention that enabled models to track the connections between words across pages, chapters and books rather than just in individual sentences. And not just words: Transformers could also use their ability to track connections to analyze code, proteins, chemicals and DNA.

  """},
    # Social Media Caption format (Let's use Twitter for this example)
    {"format": "Twitter Caption", "topic": "Beautiful Sunset", "output": "Sunsets never get old.  Feeling grateful for another stunning day. #sunset #nature"},
    # Blog Post format
    {"format": "Blog Post", "topic": "Travel Tips for Beginners", "output": """Here are some of our top travel tips for novice travelers:

  Plan your first night down to the last detail. Even if you're headed out on a free-spirited adventure, be sure that you have every detail of your first night planned. Trust us on this! Not only will you be exhausted from your travel day, but it’s good to get the lay of the land before branching out.

  Photocopy your driver’s license, passport, visa, vaccination status (if required), and other important documents. Scan them in the cloud so they're accessible wherever you are.

  Buy all your adapters, SIM cards, and currency at home. It'll be less expensive and you won't have to worry about tracking down essentials on the first day of your trip.

  Alert your credit cards and financial institutions that you will be traveling and let them know your destination. This prevents any card cancellations by your bank after making an unexpected purchase far from home. Also, inquire about details on fees and charges made overseas if traveling abroad.

  Call your phone carrier to add international service to your plan if necessary. You may also want to consider purchasing a local SIM card for the country you will be visiting.

  Working while you are away? Grab a travel Wi-Fi hotspot before you go. Be sure to add an international data plan to your regular service, too!

  Pack a travel journal. Keep a log of the places you visit to help you hold on to the memories you make.

  Pack clothes that can do double duty. Darker colors are better because they can be worn both day and night. Be sure to have a change of clothes in your carry-on, along with any other items you may need if your luggage is lost or stolen, such as medication. See more packing tips here.

  Even if you are somewhat fluent in the language spoken in the country you're visiting, bring along a phrasebook for those times when you may get stumped. This will also be helpful if your translation app isn't accessible. Also review these tips for traveling without knowing the language.

  Smile! Your first trip is something to be very excited about!

  """},
    {"format": "Press Release", "topic": "Groundbreaking AI Discovery in Medical Diagnosis", "output": 
    "FOR IMMEDIATE RELEASE\n\n[Company Name] Announces Breakthrough in AI-Powered Medical Diagnosis\n\n[City, State] – [Date] – [Company Name], a leading developer of artificial intelligence solutions, today announced a groundbreaking discovery in AI-powered medical diagnosis. Their new system, [System Name], has achieved an unprecedented level of accuracy in identifying [Specific Disease] using machine learning algorithms trained on a massive dataset of medical images and patient data. This advancement has the potential to revolutionize early detection and improve patient outcomes for millions worldwide. [Quote from company representative about the impact and future applications]"},

    # **Movie Script (Scene Excerpt) format**
    {"format": "Movie Script (Scene Excerpt)", "topic": "First Date Jitters", "output": 
    "**INT. COFFEE SHOP - DAY**\n\nEMMA (20s), dressed nervously but stylishly, waits at a table for NOAH (20s), who enters with a bouquet of flowers. He trips slightly, the flowers almost falling.\n\nEMMA\n(Smiling)\nCareful there! Almost lost your offering.\n\nNOAH\n(Chuckles, handing her the flowers)\nWouldn't want that. These are for you. They're, uh, your favorites... according to your profile.\n\nEMMA\n(Taking the flowers, surprised)\nWow, you remembered. That's sweet. Thanks.\n\nThey both sit, a moment of awkward silence hanging between them. Emma takes a sip of her coffee, trying to hide her trembling hands.\n\nNOAH\n(Clears throat)\nSo, Emma... this place is great, right? Heard they have the best lattes in town.\n\nEMMA\n(Forces a laugh)\nYeah, it's nice. Cozy.  (Beat)  So, Noah, tell me about yourself."},

    # **Instructional Manual (Excerpt) format**
    {"format": "Instructional Manual (Excerpt)", "topic": "Building a Raised Garden Bed", "output": 
    "**Step 3: Assembling the Frame**\n\n1. Lay the wooden boards flat on a clean, stable surface. Ensure all boards are the correct length and free of warping. \n2. Refer to the provided diagram for your specific raised bed design.  \n3. Pre-drill pilot holes in the ends of each board where they will be connected using screws. This prevents splitting of the wood. \n4. Using a screwdriver or power drill with appropriate attachment, secure the boards together with the designated screws. Double-check all connections for tightness. \n5. Once the frame is assembled, stand it upright on a level surface. Inspect for squareness and adjust as needed."},

    # **Song Lyrics (Chorus) format**
    {"format": "Song Lyrics (Chorus)", "topic": "Lost in the City Lights, chorus: \n(Fast tempo, driving beat)\nLost in the city lights, a million stories in the night\nSearching for a place to belong, where do I fit in this neon song?\nFaces blur, a symphony of sound\nTrying to find my way, lost and never found"},

    # **Interview Q&A format**
    {"format": "Interview Q&A", "topic": "Career Advice for Aspiring Artists", "output": 
    "Interviewer: What advice would you give to young people who dream of a career in the arts? \n\nArtist: The most important thing is passion. If you don't truly love what you create, it will be difficult to sustain the challenges of this path. But passion alone isn't enough. Develop your skills relentlessly, seek out mentors, and build a strong network. Don't be afraid to put yourself out there, even if it means facing rejection. Remember, success rarely happens overnight. Be patient, persistent, and never give up on your vision."}
  ]
  return examples




# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     # The list of examples available to select from.
#     examples,
#     # The embedding class used to produce embeddings which are used to measure semantic similarity.
#     OpenAIEmbeddings(),
#     # The VectorStore class that is used to store the embeddings and do a similarity search over.
#     Chroma,
#     # The number of examples to produce.
#     k=1,
# )
# similar_prompt = FewShotPromptTemplate(
#   # We provide an ExampleSelector instead of examples.
#   example_selector=example_selector,  # Assuming you have an example_selector defined elsewhere
#   example_prompt=example_prompt,
#   prefix="Given each format and a topic as input, respond with the text in the desired format.",
#   suffix="Topic: {topic}\nFormat: {format}\nOutput:",  # Added a newline character for better readability after the output
#   input_variables=["format", "topic"],
# )
# print(similar_prompt.format(format="Linkedin Post",topic='Generative Ai'))

class ContentAi:
    def __init__(self,format,topic):
        self.format=format
        self.topic=topic
        self.llm=OpenAI(model="gpt-3.5-turbo-instruct",temperature=1,max_tokens=526)
    def prompt_template(self):
      example_prompt = PromptTemplate(
      input_variables=["format", "topic","output"],
      template="format: {format}\ntopic: {topic}\nOutput: {output}",
      )
      examples =get_examples()
      example_selector = SemanticSimilarityExampleSelector.from_examples(
      # The list of examples available to select from.
      examples,
      # The embedding class used to produce embeddings which are used to measure semantic similarity.
      OpenAIEmbeddings(),
      # The VectorStore class that is used to store the embeddings and do a similarity search over.
      Chroma,
      # The number of examples to produce.
      k=1,
      )
      similar_prompt = FewShotPromptTemplate(
      # We provide an ExampleSelector instead of examples.
      example_selector=example_selector,  # Assuming you have an example_selector defined elsewhere
      example_prompt=example_prompt,
      prefix="Given each format and a topic as input, respond with the text in the desired format.",
      suffix="Format: {format}\nTopic: {topic}\nOutput: ",  # Added a newline character for better readability after the output
      input_variables=["format", "topic"],
      ) 
      return similar_prompt.format(format=self.format,topic=self.topic)

    def generate(self):
        llm=self.llm
        prompt=self.prompt_template()
        # print(prompt)
        return llm.invoke(prompt)

# content=ContentAi('Linkedin Post','Drawbacks of AI')
# print(content.generate())

        