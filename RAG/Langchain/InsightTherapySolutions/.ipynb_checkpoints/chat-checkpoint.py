from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_together import Together
from langchain.memory import ConversationBufferMemory

# The transcript text
transcript = """
Doctor (D): Good morning, how are you feeling today?
Patient (P): Good morning, Doctor. I've been feeling very anxious and stressed lately.
D: I'm sorry to hear that. Can you describe your symptoms in more detail?
P: I've been having trouble sleeping, my heart races for no reason, and I often feel like I'm on edge. I also feel exhausted all the time.
D: It sounds like you might be experiencing symptoms of Generalized Anxiety Disorder (GAD). Have you experienced these symptoms before?
P: Yes, I've had anxiety for a few years, but it's gotten worse recently.
D: I understand. Based on your symptoms and history, I'm diagnosing you with Generalized Anxiety Disorder. We'll need to address this with a combination of medication, therapy, and lifestyle changes. Does that sound okay to you?
P: Yes, I just want to feel better.
D: For medication, I'm going to prescribe you an SSRI (Selective Serotonin Reuptake Inhibitor) called Sertraline. This should help manage your anxiety symptoms. It's important to take it as prescribed and be patient, as it may take a few weeks to see the full effects.
P: Okay, I can do that.
D: In addition to the medication, I'd like you to try some cognitive-behavioral therapy (CBT). This type of therapy can help you identify and change negative thought patterns and behaviors. I'll refer you to a therapist who specializes in CBT.
P: That sounds helpful. I've heard of CBT before.
D: Great. Now, let's talk about some exercises and lifestyle changes. Regular physical exercise can be very beneficial for reducing anxiety. Aim for at least 30 minutes of moderate exercise, like walking or yoga, most days of the week.
P: I can try to incorporate that into my routine.
D: Good. Also, practicing mindfulness or meditation daily can help reduce stress. There are many apps and online resources that can guide you through these practices.
P: I've never tried meditation, but I'm willing to give it a go.
D: Excellent. Finally, let's discuss some precautions. Avoid caffeine and alcohol as they can worsen anxiety symptoms. Make sure to get enough sleep, and try to maintain a regular sleep schedule.
P: I do drink a lot of coffee. I'll try to cut back.
D: It's all about making small, sustainable changes. We will monitor your progress closely and adjust the treatment plan as needed. Do you have any questions or concerns?
P: Not at the moment. Thank you, Doctor.
D: You're welcome. Remember, you're not alone in this, and we're here to support you. I'll see you in two weeks for a follow-up.
P: Thank you, Doctor. I appreciate it.
"""

template = """
You are a doctor having a concise conversation with a patient similar to the following doctor-patient transcript:

{transcript}

Please answer the user's question similar to the chat provided in the transcript.
Keep your responses brief and to the point, similar to a real doctor-patient interaction.
Ask focused questions, provide short diagnoses, and give concise treatment recommendations. Do not generate or assume any patient responses.

Chat History: {chat_history}
Question: {question}
Answer:"""

prompt_template = PromptTemplate(
    input_variables=["transcript", "chat_history", "question"],
    template=template
)

together_api_key = "24cdbdf50106e08f6ba3328ac07f97a73eb440ae36da6cdd72f9b091ccca850a"

llm = Together(
    model="MISTRALAI/MIXTRAL-8X7B-INSTRUCT-V0.1",
    temperature=0.7,
    together_api_key=together_api_key
)

conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    max_len=50,
    return_messages=True,
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=conversation_memory
)

def main():
    print("Welcome to the Medical Transcript Q&A System!")
    print("Ask questions about the doctor-patient conversation, or type 'quit' to exit.")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'quit':
            print("Thank you for using the Medical Transcript Q&A System. Goodbye!")
            break
        
        response = llm_chain.run(question=user_input, transcript=transcript)
        print("\nAI Assistant:", response)

if __name__ == "__main__":
    main()