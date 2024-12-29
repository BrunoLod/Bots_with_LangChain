from prompts.contextualize_prompt import contextualize_q_prompt
from prompts.rag_system_prompt import rag_system_prompt
from prompts.system_prompt import system_prompt

HUB_PROMPTS = {
    "rag_prompt": rag_system_prompt, 
    "context_prompt": contextualize_q_prompt, 
    "system_prompt": system_prompt
}