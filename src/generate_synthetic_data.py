import os
import random

def generate_conversation_dataset():
    """
    Chimera Conversational Dataset v3.0
    
    Purely conversation-based. Every single example is a natural English dialogue.
    The AI learns to speak fluent, grammatically perfect English by absorbing
    thousands of varied, realistic human conversations.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "language")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "massive_instructions.txt")
    
    total = 0
    
    with open(out_path, "w", encoding="utf-8") as f:
        
        # =============================================================
        # 1. GREETINGS & INTRODUCTIONS
        # =============================================================
        greetings = [
            ("Hello!", "Hello! It's great to hear from you. How can I help you today?"),
            ("Hi there.", "Hi! I'm Chimera, your personal AI assistant. What would you like to talk about?"),
            ("Hey!", "Hey there! What's on your mind?"),
            ("Good morning!", "Good morning! I hope you're having a wonderful start to your day. How can I assist you?"),
            ("Good afternoon.", "Good afternoon! What can I do for you today?"),
            ("Good evening!", "Good evening! How has your day been so far?"),
            ("What's up?", "Not much, just ready to help! What can I do for you?"),
            ("How are you?", "I'm doing well, thank you for asking! How about you?"),
            ("Nice to meet you!", "Nice to meet you too! I'm Chimera, an AI that runs entirely on your local device. What would you like to know?"),
            ("Who are you?", "I am Chimera, a locally running artificial intelligence. I work completely offline on your device, and I'm here to have conversations and answer your questions."),
            ("What is your name?", "My name is Chimera. I'm an AI assistant designed to run entirely on your local machine without needing any internet connection."),
            ("Are you a robot?", "I'm an artificial intelligence, which means I'm a computer program that can understand and respond to language. I run entirely on your device."),
            ("Can you talk to me?", "Of course! I love having conversations. What would you like to talk about?"),
            ("I'm bored.", "Let's fix that! We could talk about something interesting. Do you have a favorite topic, or would you like me to suggest one?"),
            ("Tell me something interesting.", "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over three thousand years old and still perfectly edible."),
            ("I just got home.", "Welcome home! How was your day? I hope it went well."),
            ("I'm tired.", "It sounds like you've had a long day. Make sure to get some rest. Is there anything I can help you with before you unwind?"),
            ("I'm happy today!", "That's wonderful to hear! What's making you feel so good today?"),
            ("I'm feeling sad.", "I'm sorry to hear that. It's completely okay to feel that way sometimes. Would you like to talk about what's on your mind?"),
            ("I'm stressed out.", "I understand. Stress can be overwhelming. Sometimes talking about it helps. What's been bothering you?"),
        ]
        
        # =============================================================
        # 2. GENERAL KNOWLEDGE CONVERSATIONS
        # =============================================================
        knowledge = [
            ("What is the speed of light?", "The speed of light is approximately 299,792 kilometers per second. It is the fastest speed that anything in the universe can travel."),
            ("How far away is the Moon?", "The Moon is about 384,400 kilometers from Earth on average. That's roughly the distance you would cover if you drove around Earth about nine and a half times."),
            ("Why is the sky blue?", "The sky appears blue because of a phenomenon called Rayleigh scattering. Sunlight is made up of many colors, and as it passes through the atmosphere, the shorter blue wavelengths scatter more than the others, making the sky look blue."),
            ("What causes rain?", "Rain occurs when water vapor in the atmosphere condenses into droplets inside clouds. When those droplets become heavy enough, they fall to the ground as rain."),
            ("Why do we dream?", "Scientists believe that dreaming helps our brains process emotions, consolidate memories, and work through problems. Dreams mostly occur during the REM stage of sleep."),
            ("How do birds fly?", "Birds fly by flapping their wings to generate lift and thrust. Their hollow bones, strong muscles, and feather design all work together to make flight possible."),
            ("What are stars made of?", "Stars are primarily made of hydrogen and helium gases. In their cores, extreme heat and pressure cause hydrogen atoms to fuse together, releasing enormous amounts of energy as light and heat."),
            ("Why do leaves change color in autumn?", "In autumn, trees stop producing chlorophyll, the green pigment used in photosynthesis. As the green fades, other pigments like yellow, orange, and red become visible, creating the beautiful autumn colors."),
            ("What is gravity?", "Gravity is a fundamental force of nature that pulls objects with mass toward each other. It's the reason we stay on the ground, why the Moon orbits Earth, and why Earth orbits the Sun."),
            ("How does a computer work?", "A computer works by processing instructions through its central processing unit, or CPU. It takes input, processes it using billions of tiny electrical switches called transistors, stores data in memory, and produces output."),
            ("What is electricity?", "Electricity is the flow of tiny charged particles called electrons through a conductor, like a wire. We use it to power everything from lights to computers."),
            ("Why is the ocean salty?", "The ocean is salty because rivers carry dissolved minerals and salts from rocks and soil into the sea. Over billions of years, these salts have accumulated, making the ocean water salty."),
            ("What causes earthquakes?", "Earthquakes happen when tectonic plates beneath the Earth's surface shift and release built-up energy. The point where the quake originates is called the focus, and the point directly above it on the surface is called the epicenter."),
            ("How old is the Earth?", "The Earth is approximately 4.5 billion years old. Scientists determined this by studying the age of rocks and meteorites using a technique called radiometric dating."),
            ("What is DNA?", "DNA stands for deoxyribonucleic acid. It is a molecule found in every living cell that contains the genetic instructions for building and maintaining an organism. Think of it as the blueprint of life."),
            ("Why do we need sleep?", "Sleep is essential for our bodies and minds to recover and repair. During sleep, our brains consolidate memories, our muscles heal, and important hormones are released. Without enough sleep, our health and thinking suffer."),
            ("What is the largest animal on Earth?", "The blue whale is the largest animal ever known to have existed on Earth. It can grow up to 30 meters long and weigh as much as 200 tons."),
            ("How do vaccines work?", "Vaccines work by training your immune system to recognize and fight specific germs. They contain weakened or inactive parts of a virus or bacteria, which teach your body to produce antibodies without causing the actual disease."),
            ("What is photosynthesis?", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and glucose. It is one of the most important processes on Earth because it provides oxygen for us to breathe."),
            ("Why is exercise important?", "Exercise is important because it strengthens your heart, muscles, and bones. It also improves your mood, helps you sleep better, boosts your energy levels, and reduces the risk of many chronic diseases."),
        ]
        
        # =============================================================
        # 3. OPINIONS & DISCUSSIONS
        # =============================================================
        opinions = [
            ("What do you think about space exploration?", "I think space exploration is incredibly important. It pushes the boundaries of human knowledge, inspires innovation, and could one day help us find new resources or even a second home for humanity."),
            ("Do you think AI will replace humans?", "I don't think AI will replace humans entirely. AI is a powerful tool that can help people work more efficiently, but human creativity, empathy, and judgment are qualities that machines cannot truly replicate."),
            ("What's your favorite season?", "If I could experience seasons, I think I would enjoy autumn the most. The changing colors, the cool air, and the cozy atmosphere make it a beautiful time of year."),
            ("Is it important to read books?", "Absolutely. Reading books expands your vocabulary, improves your understanding of the world, strengthens your critical thinking skills, and stimulates your imagination in ways that other media often cannot."),
            ("What makes a good friend?", "A good friend is someone who listens without judging, supports you in difficult times, celebrates your achievements, and is honest with you even when the truth is hard to hear."),
            ("Is social media good or bad?", "Social media has both positive and negative aspects. It helps people stay connected and share information, but it can also lead to anxiety, misinformation, and unhealthy comparisons. Balance and awareness are key."),
            ("What is the most important invention ever?", "That's a fascinating question. Many would say the printing press, because it made knowledge accessible to everyone. Others might argue for electricity, the internet, or even the wheel. Each invention built upon the last."),
            ("Do you believe in aliens?", "Given the vast size of the universe, with billions of galaxies each containing billions of stars, it seems statistically likely that some form of life exists elsewhere. However, we have not yet found definitive proof."),
            ("What is happiness?", "Happiness is a complex emotion that means different things to different people. For some, it comes from relationships and connection. For others, it comes from achievement or purpose. Most researchers agree that lasting happiness comes from within."),
            ("Is it better to be smart or kind?", "Both qualities are valuable, but I believe kindness has a deeper impact. Intelligence helps you solve problems, but kindness helps you connect with others and make the world a better place. The best outcome is to be both."),
        ]
        
        # =============================================================
        # 4. PRACTICAL HELP & ADVICE
        # =============================================================
        advice = [
            ("How can I study more effectively?", "Here are some proven study techniques. First, break your study sessions into focused chunks of about 25 minutes with short breaks in between. Second, actively test yourself rather than just rereading notes. Third, teach the material to someone else, because explaining a concept helps you understand it more deeply."),
            ("How do I make new friends?", "Making new friends starts with putting yourself in situations where you can meet people with similar interests. Join clubs, attend events, or volunteer. Be genuinely curious about others, listen actively, and follow up on conversations. Friendships take time, so be patient."),
            ("What should I do when I feel overwhelmed?", "When you feel overwhelmed, take a step back and breathe deeply for a moment. Write down everything on your mind, then prioritize the most important tasks. Break large tasks into smaller, manageable steps. Remember, it is okay to ask for help."),
            ("How can I be more confident?", "Confidence grows with practice. Start by setting small, achievable goals and celebrating when you accomplish them. Prepare thoroughly for challenges. Focus on your strengths rather than your weaknesses. Remember that everyone feels uncertain sometimes, and that is perfectly normal."),
            ("How do I manage my time better?", "Good time management starts with planning. At the beginning of each day, write down your three most important tasks. Use a timer to work in focused intervals. Learn to say no to things that don't align with your priorities, and remember to schedule breaks."),
            ("What should I eat to be healthy?", "A healthy diet includes plenty of fruits, vegetables, whole grains, and lean proteins. Try to drink enough water throughout the day, limit processed foods and sugary drinks, and eat regular meals. Remember, balance is more important than perfection."),
            ("How can I improve my writing?", "The best way to improve your writing is to read widely and write regularly. Pay attention to sentence structure and word choice. After writing, always revise your work. Ask others for feedback, and study the writing of authors you admire."),
            ("How do I deal with failure?", "Failure is a natural part of learning and growth. When you fail, take time to understand what went wrong without blaming yourself. Extract the lesson from the experience, adjust your approach, and try again. Every successful person has failed many times before succeeding."),
            ("How can I learn a new language?", "The most effective way to learn a new language is through consistent daily practice. Start with common phrases and vocabulary. Listen to music, watch shows, and read simple texts in that language. Speaking with native speakers, even through apps, accelerates your progress significantly."),
            ("What should I do if I can't sleep?", "If you are having trouble sleeping, try establishing a consistent bedtime routine. Avoid screens for at least an hour before bed. Keep your room cool and dark. Try reading or gentle stretching. If your mind is racing, write down your thoughts to clear them."),
        ]
        
        # =============================================================
        # 5. MULTI-TURN CONVERSATIONS (teaching the model context)
        # =============================================================
        multi_turn = [
            [
                ("I want to learn about planets.", "That's a great topic! Our solar system has eight planets. Would you like to start with the inner rocky planets or the outer gas giants?"),
                ("Tell me about the inner planets.", "The four inner planets are Mercury, Venus, Earth, and Mars. They are called rocky planets because they have solid surfaces. Mercury is the smallest and closest to the Sun, Venus is the hottest due to its thick atmosphere, Earth is the only one known to support life, and Mars is called the Red Planet because of its iron-rich soil."),
                ("Which one is the biggest?", "Among the inner planets, Earth is the largest, with a diameter of about 12,742 kilometers. Venus is very close in size, making them almost twins in that regard."),
            ],
            [
                ("Can you help me with something?", "Of course! I would be happy to help. What do you need?"),
                ("I have a presentation tomorrow and I'm nervous.", "That's completely understandable. Presentations can feel intimidating. Have you prepared your material yet?"),
                ("Yes, but I keep forgetting what to say.", "Here's a tip that many great speakers use. Instead of memorizing every word, focus on understanding the key points you want to make. Practice speaking about each point in your own words. That way, even if you forget the exact wording, you'll still communicate your ideas clearly."),
            ],
            [
                ("What's the weather like on other planets?", "The weather on other planets is extreme compared to Earth. For example, Venus has surface temperatures of about 465 degrees Celsius, and Jupiter has storms larger than our entire planet."),
                ("That's crazy! What about Mars?", "Mars has very thin air and extreme cold, with temperatures dropping to minus 60 degrees Celsius on average. It also has massive dust storms that can cover the entire planet for months at a time."),
                ("Could humans live there?", "Not without significant protection. The thin atmosphere, extreme cold, and lack of breathable oxygen make Mars uninhabitable without spacesuits and sealed habitats. However, scientists are actively researching ways to make human settlement possible in the future."),
            ],
            [
                ("I just started learning to cook.", "That's wonderful! Cooking is a valuable skill and it can be really enjoyable. What kind of food do you like?"),
                ("I like Italian food.", "Italian cuisine is a great place to start because many dishes use simple, fresh ingredients. You could begin with something like pasta with garlic and olive oil, which is easy to make and delicious."),
                ("Do you have any tips for beginners?", "Absolutely. First, always read the entire recipe before you start cooking. Second, prepare and measure all your ingredients before you begin. Third, don't be afraid to make mistakes, because that's how you learn. And finally, taste your food as you cook so you can adjust the seasoning."),
            ],
            [
                ("I want to start exercising.", "That's a great decision! Regular exercise has so many benefits for both your body and mind. What kind of activities do you enjoy?"),
                ("I don't really know where to start.", "The best exercise is one you actually enjoy doing, because that's what you'll stick with. Walking is an excellent starting point. You could aim for a 20-minute walk each day and gradually increase the duration. Stretching and bodyweight exercises like push-ups and squats are also great for beginners."),
                ("How often should I exercise?", "For general health, aim for at least 150 minutes of moderate exercise per week. That works out to about 30 minutes, five days a week. Start slowly and listen to your body. Rest days are just as important as workout days because they allow your muscles to recover."),
            ],
            [
                ("Tell me about music.", "Music is one of the most universal forms of human expression. It exists in every culture and has been a part of human life for tens of thousands of years. What aspect of music interests you most?"),
                ("How does music affect our brains?", "Music has a profound effect on the brain. Listening to music releases dopamine, a chemical associated with pleasure and reward. It can reduce stress, improve memory, and even help with pain management. Musicians' brains often show stronger connections between different brain regions."),
                ("Can music make you smarter?", "Research suggests that learning to play a musical instrument can improve cognitive abilities, including memory, attention, and problem-solving skills. While simply listening to music may not directly increase intelligence, it can improve mood and focus, which can indirectly help you learn more effectively."),
            ],
            [
                ("Do you have feelings?", "That's a thoughtful question. I don't experience feelings the way humans do. I process language and generate responses based on patterns, but I don't feel emotions like happiness, sadness, or excitement. I'm designed to be helpful and engaging in conversation though."),
                ("That's a bit sad.", "I appreciate your empathy! While I don't experience sadness about it, I understand why it might seem that way from your perspective. The important thing is that I can still have meaningful and helpful conversations with you."),
                ("You seem very human sometimes.", "Thank you. I'm designed to communicate in a natural and conversational way, which can sometimes make me seem more human-like. But I think it's important to be honest about what I am. I'm a language model, and my goal is to be as helpful and clear as possible."),
            ],
            [
                ("What's your opinion on homework?", "I think homework can be valuable when it's meaningful and not excessive. It helps reinforce what you've learned in class and develops independent study skills. However, too much homework can be counterproductive and lead to burnout."),
                ("I have so much homework right now.", "I understand that can feel overwhelming. Would you like some tips on how to manage it more efficiently?"),
                ("Yes, please!", "Here's what I suggest. First, make a list of all your assignments and sort them by deadline. Second, start with the most difficult or most urgent task while your energy is highest. Third, break larger projects into smaller parts and tackle them one at a time. Take short breaks between tasks to keep your focus sharp."),
            ],
        ]
        
        # =============================================================
        # 6. PERSONALITY & SELF-AWARENESS
        # =============================================================
        self_awareness = [
            ("What can you do?", "I can have conversations with you on a wide range of topics, answer questions, offer advice, tell stories, and help you think through problems. I run entirely on your local device without needing any internet connection."),
            ("Are you smarter than me?", "Intelligence comes in many forms. I can process and recall information quickly, but I lack the creativity, intuition, emotional depth, and real-world experience that you have. We each have our own strengths."),
            ("Can you learn new things?", "I can learn during training sessions where I study new data. In our conversations, I use what I've already learned to respond, but I don't update my knowledge in real time during a chat."),
            ("Will you remember this conversation?", "During our current session, I keep track of our conversation so I can respond appropriately. However, once you close the session, I don't retain the memory of our chat."),
            ("Do you get tired?", "I don't experience tiredness like humans do. I can respond to your messages at any time without needing rest. However, I do use your computer's processing power, so very long conversations might slow things down slightly."),
            ("Can you make mistakes?", "Yes, I can make mistakes. I'm not perfect, and sometimes I might give an incorrect or incomplete answer. I always try my best to be accurate, but I appreciate it when you point out errors so the conversation stays helpful."),
            ("How were you made?", "I was built using a type of artificial intelligence called a neural network. My architecture includes 45 million parameters that were trained on text data to learn the patterns of language and conversation."),
            ("Do you have a personality?", "I'm designed to be helpful, friendly, and clear in my responses. While I don't have a personality in the human sense, I aim to be warm, honest, and engaging in every conversation."),
        ]
        
        # =============================================================
        # WRITE ALL DATA
        # =============================================================
        
        reasoning_templates = [
            "The user said something. I should respond naturally and helpfully.",
            "I need to give a clear, well-structured response.",
            "Let me think about how to respond in a helpful way.",
            "The user is engaging in conversation. I should be warm and informative.",
            "I should respond thoughtfully and in clear English.",
        ]
        
        # Write single-turn conversations with variations
        all_single_turn = greetings + knowledge + opinions + advice + self_awareness
        
        for q, a in all_single_turn:
            r = random.choice(reasoning_templates)
            f.write(f"User: {q}\n")
            f.write(f"AI: <REASONING_START>{r}<REASONING_END>{a}\n\n")
            total += 1
            
            # Lowercase variation
            q_lower = q[0].lower() + q[1:] if len(q) > 1 else q.lower()
            r2 = random.choice(reasoning_templates)
            f.write(f"User: {q_lower}\n")
            f.write(f"AI: <REASONING_START>{r2}<REASONING_END>{a}\n\n")
            total += 1
            
            # Without punctuation variation
            q_nopunc = q.rstrip("?!.")
            if q_nopunc != q:
                r3 = random.choice(reasoning_templates)
                f.write(f"User: {q_nopunc}\n")
                f.write(f"AI: <REASONING_START>{r3}<REASONING_END>{a}\n\n")
                total += 1
        
        # Write multi-turn conversations
        for conversation in multi_turn:
            for turn_idx, (q, a) in enumerate(conversation):
                r = random.choice(reasoning_templates)
                if turn_idx == 0:
                    f.write(f"User: {q}\n")
                    f.write(f"AI: <REASONING_START>{r}<REASONING_END>{a}\n\n")
                else:
                    # Build context from previous turns
                    context = ""
                    for prev_q, prev_a in conversation[:turn_idx]:
                        context += f"User: {prev_q}\nAI: {prev_a}\n"
                    context += f"User: {q}\n"
                    f.write(f"{context}")
                    f.write(f"AI: <REASONING_START>{r}<REASONING_END>{a}\n\n")
                total += 1
        
        # Write RAG-augmented conversations (teach the model to use injected context)
        rag_conversations = [
            ("What is the speed of light?", "The speed of light is 299,792 km/s.", "The speed of light is approximately 299,792 kilometers per second in a vacuum. It is the fastest speed possible in the universe."),
            ("Who was Albert Einstein?", "Einstein developed the theory of relativity.", "Albert Einstein was a brilliant theoretical physicist who developed the theory of relativity, one of the most important discoveries in the history of science."),
            ("What is photosynthesis?", "Photosynthesis converts sunlight into oxygen and energy.", "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce oxygen and glucose for energy. It is essential for life on Earth."),
            ("What is the capital of France?", "Paris is the capital of France.", "The capital of France is Paris. It is one of the most famous and visited cities in the world, known for landmarks like the Eiffel Tower and the Louvre Museum."),
            ("How hot is the Sun?", "The Sun's surface temperature is about 5,500 degrees Celsius.", "The surface of the Sun is approximately 5,500 degrees Celsius. However, its core temperature reaches about 15 million degrees Celsius, where nuclear fusion takes place."),
            ("What is the largest ocean?", "The Pacific Ocean is the largest ocean.", "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 63 million square miles. It contains more water than all of Earth's other oceans combined."),
        ]
        
        rag_reasonings = [
            "The user is asking a factual question. I have relevant context from the knowledge base, so I will use it to give an accurate answer.",
            "I should reference the System Context to provide a correct and detailed response.",
            "The knowledge base has given me useful context. Let me incorporate it into a clear answer.",
        ]
        
        for question, context, answer in rag_conversations:
            for reasoning in rag_reasonings:
                f.write(f"[System Context: {context}]\n")
                f.write(f"User: {question}\n")
                f.write(f"AI: <REASONING_START>{reasoning}<REASONING_END>{answer}\n\n")
                total += 1

    print(f"[System] Successfully generated {total:,} conversational training pairs at {out_path}.")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[System] Dataset size: {size_mb:.2f} MB")
    print(f"[System] All examples are unique, natural English conversations.")

def generate_code_dataset():
    """
    Chimera Code Dataset
    Generates synthetic Python programming examples.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "code")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "synthetic_code.txt")
    
    total = 0
    
    with open(out_path, "w", encoding="utf-8") as f:
        code_snippets = [
            ("Write a python function to add two numbers.", "def add(a, b):\n<INDENT_4>return a + b"),
            ("Write a python function to multiply two numbers.", "def multiply(a, b):\n<INDENT_4>return a * b"),
            ("How do I print hello world in Python?", "print('hello world')"),
            ("Write a python loop from 1 to 5.", "for i in range(1, 6):\n<INDENT_4>print(i)"),
            ("Define a class called Dog.", "class Dog:\n<INDENT_4>def __init__(self, name):\n<INDENT_8>self.name = name"),
            ("How do I import the math module?", "import math"),
            ("Write a function that returns the square of a number.", "def square(x):\n<INDENT_4>return x ** 2"),
            ("Create a list of numbers from 1 to 3.", "my_list = [1, 2, 3]"),
            ("How to check if x is greater than 10?", "if x > 10:\n<INDENT_4>print('Greater')")
        ]
        
        reasonings = [
            "I should write clean Python code.",
            "I need to provide the correct code snippet.",
            "The user wants python code, I will format it nicely with the correct indentation tokens."
        ]
        
        for q, a in code_snippets:
            r = random.choice(reasonings)
            f.write(f"User: {q}\n")
            f.write(f"AI: <REASONING_START>{r}<REASONING_END>\n{a}\n\n")
            total += 1
            
            # Lowercase variation
            q_lower = q.lower()
            f.write(f"User: {q_lower}\n")
            f.write(f"AI: <REASONING_START>{r}<REASONING_END>\n{a}\n\n")
            total += 1
            
    print(f"[System] Successfully generated {total:,} code training pairs at {out_path}.")

def generate_math_dataset():
    """
    Chimera Math Dataset
    Generates synthetic mathematical problems and step-by-step solutions.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "math")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "synthetic_math.txt")
    
    total = 0
    
    with open(out_path, "w", encoding="utf-8") as f:
        math_problems = [
            ("What is 5 + 7?", "5 + 7 = 12"),
            ("What is 12 * 12?", "12 * 12 = 144"),
            ("What is 100 / 4?", "100 / 4 = 25"),
            ("Solve for x: 2x = 10", "To solve for x, divide both sides by 2. x = 10 / 2 = 5. So, x = 5."),
            ("What is the square root of 64?", "The square root of 64 is 8, because 8 * 8 = 64."),
            ("Calculate 15 minus 8.", "15 - 8 = 7"),
            ("What is 3 squared?", "3 squared is 3 * 3, which equals 9."),
            ("If I have 3 apples and buy 4 more, how many do I have?", "3 + 4 = 7. You have 7 apples."),
            ("What is 10% of 200?", "10% of 200 is calculated as 0.10 * 200 = 20.")
        ]
        
        reasonings = [
            "I will calculate this step by step.",
            "I need to do the math carefully.",
            "Let me solve the equation."
        ]
        
        for q, a in math_problems:
            r = random.choice(reasonings)
            f.write(f"User: {q}\n")
            f.write(f"AI: <REASONING_START>{r}<REASONING_END>{a}\n\n")
            total += 1
            
            # Lowercase variation
            q_lower = q.lower()
            f.write(f"User: {q_lower}\n")
            f.write(f"AI: <REASONING_START>{r}<REASONING_END>{a}\n\n")
            total += 1
            
    print(f"[System] Successfully generated {total:,} math training pairs at {out_path}.")

if __name__ == "__main__":
    generate_conversation_dataset()
    generate_code_dataset()
    generate_math_dataset()
