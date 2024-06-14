### ORIGINAL_assembly_context_prompt_train.json
prompt = f"Sequence type: {toy}\n"
prompt += f"Input Sequence:\n {', '.join(map(str,hist))}\n"
prompt += f"Next Symbol:\n {next_sym}\n---\n"

### CONTEXT-INPUT-OUTPUT_assembly_context_prompt_train.json
prompt = f"Context: {toy}\n"
prompt += f"Input:\n {', '.join(map(str,hist))}\n"
prompt += f"Output:\n {next_sym}\n---\n"

### GIVEN-THIS--COMPLETE-THIS_assembly_context_prompt_train.json
prompt = f"Given the sequences of the following: {toy}\n"
prompt += f"Complete the following sequence:\n {', '.join(map(str,hist))}\n"
prompt += f":\n {next_sym}\n---\n"
