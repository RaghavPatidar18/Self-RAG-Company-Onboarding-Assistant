from database import get_long_term_memories

USER_ID = "12345"

for memory in get_long_term_memories(USER_ID):
    print(memory)
