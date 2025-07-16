# Project Evaluation Report

This document outlines the methods used to evaluate the Corporate Intelligence Chatbot, along with the findings from these tests.

## 1. Qualitative Analysis: Logging LLM Interactions

To understand the internal workings of the RAG (Retrieval-Augmented Generation) system, a dedicated script, `evaluate.py`, was created.

### Objective
The primary objective of this script is to log the "LLM interaction" for each query. This means we can see the exact context chunks retrieved from the Pinecone database before they are passed to the Llama 3 model. This is crucial for debugging and verifying the retriever's performance.

### How to Use the Logging Tool
An assessor can verify the retrieval process by following these steps:
1.  Ensure all dependencies are installed and the `.env` file is configured.
2.  Activate the virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`).
3.  Run the evaluation script from the terminal:
    ```bash
    python evaluate.py
    ```
4.  Ask a question at the prompt. The terminal will output a detailed log.

### Example Log Output
Here is an example of what the log looks like for the question, "What is Qubits?":


--- RETRIEVED CONTEXT ---

[CHUNK 1 from Transforming-research-with-quantum-compu_2026_Journal-of-Economy-and-Technol.pdf]:
Traditional systems are essentially faster versions of basic digital devices, which only handle one “bit” of data, a binary 1 or 0. Like an on/off switch, 0 means “off” and 1 means “on”. Conventional computers employ millions of bits, each representing a 0 or 1, to do any task. Nevertheless, quantum devices employ “qubits” instead of bits. Qubits can be any floating point number between 0 and 1, inclusive of both, thanks to quantum mechanics (Procopio et al., 2015). They can coexist or be present at any time. At the subatomic scale, quantum technology exploits the unique feature that quantum particles may be in several states at once (called “superposition”). Quantum techniques also make use of “entanglement”, the next basic characteristic. Unlike conventional bits that assign bit values independently, qubits allow the arrangement of bits in an entangled state (Rab et al., 2017). While two entangled qubits, although physically separated, may maintain an associated global state, that

[CHUNK 2 from Transforming-research-with-quantum-compu_2026_Journal-of-Economy-and-Technol.pdf]:
8. Cybersecurity: Organisations will use quantum-enhanced cybersecurity to defend their digital assets and infrastructure. These include quantum-resistant encryption and key distribution (Bernstein and Lange, 2017). Quantum computing-enhanced cyber- security can help companies prepare for emerging threats in today’s digital environment. A quantum machine may resolve mathematical inquiries in microseconds; hence, a post-quantum encryption system protects traditional data encryption (MacQuarrie et al., 2020). The purpose of post-quantum digital technology was to safeguard the foundations and techniques of symmetrical cryptography against quantum attacks. Current commercial quantum machines cannot replace traditional super- computers due to the difficulty in scaling up the amount of qubits achieved at this point.

[CHUNK 3 from Transforming-research-with-quantum-compu_2026_Journal-of-Economy-and-Technol.pdf]:
and data? How can current commercial quantum devices replace supercomputers by scaling up qubits? How might quantum technology facilitate the efficient management of large datasets? How can we use quantum computing to enhance the efficiency of transport networks and reduce delays?

--------------------------------------------------------------------------------
LLM ANSWER: According to the provided context, qubits are the fundamental units of quantum information in quantum devices. Unlike traditional bits, which can only be in one of two states (0 or 1), qubits can exist in multiple states simultaneously, known as "superposition". This means that a qubit can represent any floating-point number between 0 and 1, inclusive of both.

In addition, qubits can also be in an "entangled" state, which allows them to be connected in such a way that the state of one qubit is dependent on the state of the other, even if they are physically separated. This property of qubits is unique to quantum mechanics and is not found in traditional computing systems.

In summary, qubits are the quantum equivalent of bits, but with the ability to exist in multiple states simultaneously and be entangled with other qubits, allowing for new and powerful forms of computation and information processing.
================================================================================