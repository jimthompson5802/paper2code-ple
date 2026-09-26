---
title: Security Review Findings
description: Static security review of Paper2Code and s2orc-doc2json, including external communications and prioritized risks
ms.date: 2026-09-26
ms.topic: reference
keywords:
  - security review
  - Paper2Code
  - s2orc-doc2json
---

## Executive summary

The static review found no obvious covert malware, persistence mechanism, or hidden telemetry in the first-party source examined. It did identify expected external AI/model-service traffic in Paper2Code and significant web-input risks in s2orc-doc2json, especially server-side request forgery (SSRF), unauthenticated service exposure, unsafe upload-path handling, and resource exhaustion.

This review covers source code, scripts, configuration, and dependency manifests in the workspace copies of Paper2Code and s2orc-doc2json. It does not include runtime traffic capture, a full dependency vulnerability scan, or an audit of downloaded model files, third-party package source, Grobid binaries, or container contents. Paths below are relative to the named repository root.

## External communications

### Paper2Code

* OpenAI API calls use the `OPENAI_API_KEY` environment variable. The code does not hardcode the key. OpenAI-mode planning and coding include paper JSON or LaTeX content in prompts. The debugging stage sends generated repository files and execution-error text. See `Paper2Code/codes/1_planning.py:20,49-53,218-224`, `Paper2Code/codes/3_coding.py:22,34-37,77,138-144`, and `Paper2Code/codes/4_debugging.py:126,170-185,214-225,248-251`.
* `Paper2Code/codes/1.2_rag_config.py:87,118-129` queries Hugging Face for model metadata. The local-model workflow downloads tokenizer/model files from Hugging Face through Transformers and vLLM. See `Paper2Code/codes/1_planning_llm.py:225-242`.
* Python package installation contacts the configured package index. `Paper2Code/requirements.txt` includes dependencies without exact version pins.

These are explicit workflow behaviors, not covert exfiltration. Use the OpenAI workflow only when the paper, source code, and error data may be sent to the selected provider. The OpenAI client is not given a custom base URL in the reviewed code.

### s2orc-doc2json

* The Flask `/upload_url` handler makes a server-side HTTP request to the caller-supplied URL. See `s2orc-doc2json/doc2json/flask/app.py:56-64`.
* PDF processing sends the complete PDF to the configured Grobid server over plain HTTP. Its default is `localhost:8070`, but the host is configurable. See `s2orc-doc2json/doc2json/grobid2json/grobid/grobid_client.py:20-22,67-111`.
* The Grobid configuration selects Crossref for consolidation and includes a Glutton service URL. In the doc2json client, consolidation flags default to false, so actual third-party metadata lookups depend on server configuration and request options. See `s2orc-doc2json/doc2json/grobid2json/grobid/grobid.yaml:27-32` and `s2orc-doc2json/doc2json/grobid2json/grobid/grobid_client.py:24-30`.
* Setup downloads Grobid from GitHub and Docker Compose pulls a Grobid image. These are setup-time supply-chain dependencies. See `s2orc-doc2json/scripts/setup_grobid.sh:7-11` and `s2orc-doc2json/docker-compose.yml:3-7`.

No real API key or password was found in the reviewed configuration. The Crossref token shown in configuration is a placeholder.

## Findings

### High: SSRF through the URL upload endpoint

The `/upload_url` route fetches a caller-controlled URL without visible scheme/host allowlisting, private-network filtering, response-size limits, or a timeout. A remote caller may use it to probe or access services reachable from the server, including internal network resources, and large responses can consume memory. See `s2orc-doc2json/doc2json/flask/app.py:56-64`.

Remove the route if it is not required. Otherwise, allow only approved hosts and HTTP(S), reject loopback, private, link-local, and metadata-service destinations after DNS resolution and redirects, and enforce connect/read timeouts plus response-size limits.

### High: Unauthenticated services may be reachable from the network

The Flask development server binds to `0.0.0.0` on port 8080. Compose publishes Grobid ports 8070 and 8071 without binding them to loopback. If these defaults are used on a network-reachable host, unauthenticated users can submit work and consume resources. Grobid's configured CORS origin is also `*`. See `s2orc-doc2json/doc2json/flask/app.py:68`, `s2orc-doc2json/docker-compose.yml:3-7`, and `s2orc-doc2json/doc2json/grobid2json/grobid/grobid.yaml:47`.

Bind services to loopback for local use. For shared deployment, use a production WSGI server, authentication, network restrictions, and a reverse proxy with request limits. Restrict published ports to loopback unless external access is intentional.

### High: Uploaded filenames can write outside the intended temp directory

The LaTeX and JATS stream handlers join the client-provided filename to a temp directory and write the uploaded bytes without first sanitizing the name or verifying path containment. A crafted filename may cause writes to other paths accessible to the service account. See `s2orc-doc2json/doc2json/tex2json/process_tex.py:27-38` and `s2orc-doc2json/doc2json/jats2json/process_jats.py:22-36`.

Discard client path components, generate server-side temporary names, and verify the resolved destination remains within the designated upload directory.

### High: Archive extraction path check is not a safe containment check

Tar extraction checks paths with `os.path.commonprefix`, which compares strings rather than path components. Similar-prefix paths can pass this check while referring outside the extraction directory. The project documents Python 3.8 support, where newer standard-library extraction protections may not be present. The gzip path also reads the fully decompressed stream into memory. See `s2orc-doc2json/doc2json/tex2json/tex_to_xml.py:47-66,68-96`.

Use a supported safe extraction filter or a component-aware containment check such as `os.path.commonpath`, reject links and special files, and impose compressed and expanded size limits. Test the behavior on every supported Python version.

### High: Model-controlled file paths in Paper2Code

Paper2Code writes generated files using model-derived task-list paths without checking that resolved paths remain under `output_repo_dir`. A manipulated paper or model response could direct writes elsewhere. See `Paper2Code/codes/3_coding.py:221` and `Paper2Code/codes/3_coding_llm.py:260`.

Resolve each destination and reject it unless it is contained beneath the configured output root. Apply the same containment rule to any model-generated path before creating directories or files.

### High: Remote model code is explicitly trusted and executed

The local-model scripts set `trust_remote_code=True` when loading selected Qwen and DeepSeek models. This allows downloaded model repositories to execute Python code locally. See `Paper2Code/codes/1_planning_llm.py:225-242`, `Paper2Code/codes/2_analyzing_llm.py:150-167`, and `Paper2Code/codes/3_coding_llm.py:158-175`.

Disable remote-code trust where possible. Otherwise, use only reviewed repositories, pin an immutable revision, and run inference with least privilege in an isolated environment.

### Conditional: Debug patch paths are not confined to the repository

If reached, the patch applier uses filenames from the OpenAI response with `os.path.join(debug_dir, filename)`. Absolute filenames can bypass `debug_dir`; existing files are renamed and rewritten when a patch matches. However, the current CLI appears to fail earlier: `args.output_repo_dir` is accessed but is not declared in `parse_args()`. Treat the path issue as latent unless that invocation path is repaired. See `Paper2Code/codes/4_debugging.py:11-27,65-74,83-122,138,248-251`.

When making this path reachable, require resolved patch paths to stay under the selected repository and require a human review before applying model-generated changes.

### Medium: Unbounded document processing can exhaust resources

The Flask upload handler reads submitted PDFs, archives, and XML into memory. No request-size or processing quotas are visible. Combined with archive expansion and expensive parsing, this can enable denial of service. See `s2orc-doc2json/doc2json/flask/app.py:22-46` and `s2orc-doc2json/doc2json/tex2json/tex_to_xml.py:68-71`.

Set upload, decompression, time, concurrency, and temporary-storage limits. Do not expose the Flask development server directly to untrusted clients.

### Medium: Legacy and broadly ranged dependencies

The s2orc-doc2json manifest pins older packages, including Flask 1.0.2, while other dependencies use broad or absent version constraints. Paper2Code also lacks exact pins for several packages. This is a supply-chain and patch-management risk, not evidence of malware. See `s2orc-doc2json/requirements.txt:1-7` and `Paper2Code/requirements.txt`.

Upgrade dependencies to supported releases, create a reproducible lock file, and run a package vulnerability scan before deployment.

## Overall assessment

The primary confidentiality decision is whether Paper2Code may send full papers, code, and errors to an external model provider. The primary deployment risks are s2orc-doc2json's public-facing URL fetch and upload routes, filesystem path handling, and unconstrained document processing. No source changes were made as part of this review.
