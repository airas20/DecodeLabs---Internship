def get_response(user_input):
    text = user_input.lower().strip()

    
    if text in ["hi", "hello", "hey"]:
        return "Hello! I am RoboBot. How can I help you?"

    # Name
    elif "your name" in text or "who are you" in text:
        return "My name is RoboBot. I am a rule-based AI chatbot."

    
    elif "how are you" in text:
        return "I'm doing great! Thanks for asking."

    # Artificial Intelligence
    elif "what is ai" in text or "artificial intelligence" in text:
        return "AI is the simulation of human intelligence by computer systems."

   
    elif "project" in text:
        return "This is Project 1: a Rule-Based AI Chatbot built using Python and if-else logic."

    # Help
    elif "help" in text:
        return "You can ask me about AI, my name, or this project. Type 'bye' to exit."

    
    elif "thank" in text:
        return "You're welcome!"

    
    elif text in ["bye", "exit", "quit", "goodbye"]:
        return "Goodbye! Have a great day!"

    
    else:
        return "Sorry, I don't understand that. Try asking me something else."


def chatbot():

    print("=" * 50)
    print("          ROBObot - AI CHATBOT")
    print("=" * 50)

    print("Hello! I am RoboBot, a rule-based AI chatbot.")
    print("Type 'help' for available commands.")
    print("Type 'bye' to exit.")
    print("-" * 50)

    while True:

        user_input = input("You: ")

        response = get_response(user_input)

        print("RoboBot:", response)

        if user_input.lower().strip() in ["bye", "exit", "quit", "goodbye"]:
            break


if __name__ == "__main__":
    chatbot()