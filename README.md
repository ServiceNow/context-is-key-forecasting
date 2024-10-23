# Context is Key: A Benchmark for Forecasting with Essential Textual Information

This repository hosts the benchmark visualization website at **https://servicenow.github.io/context-is-key-forecasting/** supporting the following paper:

**Context is Key: A Benchmark for Forecasting with Essential Textual Information**.

Preprint (2024).
_Andrew Robert Williams*, Arjun Ashok*, Étienne Marcotte, Valentina Zantedeschi, Jithendaraa Subramanian, Roland Riachi, James Requeima, Alexandre Lacoste, Irina Rish, Nicolas Chapados, Alexandre Drouin_.


> Forecasting is a critical task in decision making across various domains. While numerical data provides a foundation, it often lacks crucial context necessary for accurate predictions. Human forecasters frequently rely on additional information, such as background knowledge or constraints, which can be efficiently communicated through natural language. However, the ability of existing forecasting models to effectively integrate this textual information remains an open question. 
To address this, we introduce "Context is Key" (CiK), a time series forecasting benchmark that pairs numerical data with diverse types of carefully crafted textual context, requiring models to integrate both modalities. 
We evaluate a range of approaches, including statistical models, time series foundation models, and LLM-based forecasters, and propose a simple yet effective LLM prompting method that outperforms all other tested methods on our benchmark. 
Our experiments highlight the importance of incorporating contextual
information, demonstrate surprising performance when using LLM-based forecasting models, and also reveal some of their critical shortcomings. 
By presenting this benchmark, we aim to advance multimodal forecasting, promoting models that are both accurate and accessible to decision-makers with varied technical expertise.
The benchmark can be visualized at https://servicenow.github.io/context-is-key-forecasting/.

## License

This code is released under an Apache-2 license. Please see [terms](./LICENSE).