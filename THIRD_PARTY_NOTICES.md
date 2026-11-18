# Third-party components

This repository contains original application and documentation code under the MIT license. Dependencies retain their own licenses; no vendor checkout or node_modules is bundled. The attributed Python patch parser adaptation below is included.

- [NOOA / NVIDIA NeMo Labs OO Agents](https://github.com/NVIDIA-NeMo/labs-OO-Agents), pinned in pyproject.toml and uv.lock. Upstream Apache-2.0; native strategies, context, storage, channels and admission mechanisms are attributed to that project.
  At runtime, two pinned NOOA guard modules are copied into a host-owned bootstrap directory with their Apache-2.0 headers retained; only the config import is redirected to avoid importing the full SDK in scientific children. These generated private files are not published.
- [DeepSeek Harness fs-patch](https://github.com/deepseek-ai/deepseek-harness/tree/39163dd30f703a79079e2d2446545cff3c453738/packages/fs/fs-patch), commit `39163dd30f703a79079e2d2446545cff3c453738`: patch parsing and text matching are adapted into Python in `src/factor_demo/autoresearch_agent_v2/tools/builtin_tools/files/patch.py`. DeepSeek contributions are MIT; the adapted OpenAI Codex portions are Apache-2.0. The complete upstream notices, licenses and local modification notice are retained in the adjacent `PATCH_LICENSE`. The filesystem execution adapter is local code.
- [DeepSeek Harness compaction-basic](https://github.com/deepseek-ai/deepseek-harness/blob/39163dd30f703a79079e2d2446545cff3c453738/packages/compaction/compaction-basic/src/summarizer.ts)，提交 `39163dd30f703a79079e2d2446545cff3c453738`：`agent/history.py` 沿用其英文压缩指令、摘要引导语与标签，并适配现有 Python 会话和模型调用。上游版权和 MIT 许可保存在 `src/factor_demo/autoresearch_agent_v2/agent/COMPACTION_LICENSE`。
- Python scientific stack: NumPy, SciPy, pandas, Polars, scikit-learn, aeon, statsmodels, PyArrow, Matplotlib and transitive dependencies. See the locked package distributions for their respective licenses.
- FastAPI, Uvicorn, Pydantic, httpx and psutil provide service/runtime infrastructure.
- Vue, Vite, TypeScript, marked, DOMPurify and lucide provide the frontend.
- Sphinx, Furo, MyST Parser, sphinxcontrib-mermaid and Mermaid CLI build the documentation.

Lam Research and Applied Materials product names identify public reference materials only. This project is not affiliated with those manufacturers and does not include their proprietary recipes, source code, manuals or calibrated equipment parameters. Literature references are methodological attribution, not a claim of reproducing those research systems.
