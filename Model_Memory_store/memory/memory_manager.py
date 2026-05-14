from config import SUPABASE_KEY,SUPABASE_URL
from supabase import create_client
from datetime import datetime, timezone


class MemoryLayer:
    def __init__(self, supabase):
        self.supabase = supabase
    
    def create_user(self,user_name:str, user_email:str):
        try:
            response = self.supabase.table("USERS").insert({
                    "name": user_name,
                    "email": user_email
                }).execute()
            return response
        except Exception as e:
            print(f"Error getting user: {e}")
            raise
        
    def get_user(self, user_id:str):
        try:
            response = self.supabase.table("USERS").select("*").eq("id", user_id).execute()
            return response.data
        except Exception as e:
            print(f"Error getting user: {e}")
            raise

    def create_conversation(self,user_id:str,title:str):
        try: 
            response =self.supabase.table("CONVERSATIONS").insert({
                "user_id":user_id, "title":title
                }).execute() 
            return response
        except Exception as e:
            print(f"Error create_conversation: {e}")
            raise
        
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict = None):
        try:
            response = self.supabase.table("MESSAGES").insert({
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},          
                "created_at": datetime.utcnow().isoformat() 
            }).execute()
            return response
        except Exception as e:
            print(f"Error adding message: {e}")
            raise
    
    def get_recent_messages(self, conversation_id: str, k: int):
        try:
            response = (
                self.supabase.table("MESSAGES")
                .select("*")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=True)  
                .limit(k)                         
                .execute()
            )
            return list(reversed(response.data)) if response.data else []
        except Exception as e:
            print(f"Error fetching messages: {e}")
            raise
    
    
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

memory= MemoryLayer(supabase)


# id="1cede12c-6dbf-4db2-a6f3-07472573c987"
# title="Alok in this line"
# conv_id="4cfaf3fb-4a3d-4d0d-9147-70da25ed485d"

# print(memory.create_conversation(id,title))

metadata={"sources":"Testing for supabase","confidence_score":0.98,'rank':1}
# print(memory.add_message(conv_id,role="User",content="Hi how are you",metadata=metadata))
# history=memory.get_recent_messages(conv_id,4)
# print(history)



# memory_context = "\n".join(
#     f"{msg['role'].capitalize()}: {msg['content'][:200]}..."  # limit length
#     for msg in history
# ) if history else None

# context="document context"
# query="what do we learn"
# prompt = f"""
# You are a helpful assistant. Answer the user's question using the document context below as your primary source. You may also use any relevant previous conversation memory provided. If neither the context nor the memory contains enough information, say "I don't know" clearly.
# --- DOCUMENT CONTEXT ---
# {context}
# --- END DOCUMENT CONTEXT ---
# --- PREVIOUS MEMORY ---
# {memory_context if memory_context else "No previous memory available."}
# --- END PREVIOUS MEMORY ---
# User Question: {query}
# Answer:
#     """
# print(prompt)
# python -m Model_Memory_store.memory.memory_manager