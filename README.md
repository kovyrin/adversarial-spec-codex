# adversarial-spec

A Codex-first skill and CLI that iteratively refines product specifications through multi-model debate until consensus is reached.

**Key insight:** A single LLM reviewing a spec will miss things. Multiple LLMs debating a spec will catch gaps, challenge assumptions, and surface edge cases that any one model would overlook. The result is a document that has survived rigorous adversarial review.

**Codex is an active participant**, not just an orchestrator. Codex provides independent critiques, challenges opponent models, and contributes substantive improvements alongside external models.

## Quick Start (Codex)

1. Copy or symlink `skills/adversarial-spec` into `~/.codex/skills/adversarial-spec`
2. Set at least one API key (e.g., `OPENAI_API_KEY`)
3. Ask Codex: “Use adversarial-spec to draft a PRD for <your idea>”

## Quick Start (CLI)

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Point to the debate script
export DEBATE_PY="$PWD/skills/adversarial-spec/scripts/debate.py"
# Or, if installed as a Codex skill:
# export DEBATE_PY="$HOME/.codex/skills/adversarial-spec/scripts/debate.py"

# 3. Set at least one API key
export OPENAI_API_KEY="sk-..."
# Or use OpenRouter for access to multiple providers with one key
export OPENROUTER_API_KEY="sk-or-..."

# 4. Run it
python3 "$DEBATE_PY" critique --models gpt-4o --doc-type tech <<'SPEC_EOF'
<your spec here>
SPEC_EOF
```

## How It Works

```
You describe product --> Codex drafts spec --> Multiple LLMs critique in parallel
        |                                              |
        |                                              v
        |                              Codex synthesizes + adds own critique
        |                                              |
        |                                              v
        |                              Revise and repeat until ALL agree
        |                                              |
        +--------------------------------------------->|
                                                       v
                                            User review period
                                                       |
                                                       v
                                            Final document output
```

1. Describe your product concept or provide an existing document
2. (Optional) Start with an in-depth interview to capture requirements
3. Codex checks `CONSTITUTION.md` in the project root and uses it to scope assumptions
4. Codex drafts the initial document (PRD or tech spec)
5. Document is sent to opponent models (GPT, Gemini, Grok, etc.) for parallel critique
6. Codex provides independent critique alongside opponent feedback
7. Codex synthesizes all feedback and revises
8. Loop continues until ALL models AND Codex agree
9. User review period: request changes or run additional cycles
10. Final converged document is output

## Requirements

- Python 3.10+
- `litellm` package: `pip install litellm`
- API key for at least one LLM provider

## Supported Models

| Provider   | Env Var                | Example Models                               |
|------------|------------------------|----------------------------------------------|
| OpenAI     | `OPENAI_API_KEY`       | `gpt-4o`, `gpt-4-turbo`, `o1`                |
| Anthropic  | `ANTHROPIC_API_KEY`    | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| Google     | `GEMINI_API_KEY`       | `gemini/gemini-2.0-flash`, `gemini/gemini-pro` |
| xAI        | `XAI_API_KEY`          | `xai/grok-3`, `xai/grok-beta`                |
| Mistral    | `MISTRAL_API_KEY`      | `mistral/mistral-large`, `mistral/codestral` |
| Groq       | `GROQ_API_KEY`         | `groq/llama-3.3-70b-versatile`               |
| OpenRouter | `OPENROUTER_API_KEY`   | `openrouter/openai/gpt-4o`, `openrouter/anthropic/claude-3.5-sonnet` |
| Codex CLI  | ChatGPT subscription   | `codex/gpt-5.3-codex`, `codex/gpt-5.2-codex` |
| Gemini CLI | Google account         | `gemini-cli/gemini-3-pro-preview`, `gemini-cli/gemini-3-flash-preview` |
| Deepseek   | `DEEPSEEK_API_KEY`     | `deepseek/deepseek-chat`                     |
| Zhipu      | `ZHIPUAI_API_KEY`      | `zhipu/glm-4`, `zhipu/glm-4-plus`            |

Check which keys are configured:

```bash
python3 "$DEBATE_PY" providers
```

## AWS Bedrock Support

For enterprise users who need to route all model calls through AWS Bedrock (e.g., for security compliance or inference gateway requirements):

```bash
# Enable Bedrock mode
python3 "$DEBATE_PY" bedrock enable --region us-east-1

# Add models enabled in your Bedrock account
python3 "$DEBATE_PY" bedrock add-model claude-3-sonnet
python3 "$DEBATE_PY" bedrock add-model claude-3-haiku

# Check configuration
python3 "$DEBATE_PY" bedrock status

# Disable Bedrock mode
python3 "$DEBATE_PY" bedrock disable
```

When Bedrock is enabled, **all model calls route through Bedrock** - no direct API calls are made. Use friendly names like `claude-3-sonnet` which are automatically mapped to Bedrock model IDs.

Configuration is stored at `~/.config/adversarial-spec/config.json`.

## OpenRouter Support

[OpenRouter](https://openrouter.ai) provides unified access to multiple LLM providers through a single API. This is useful for:
- Accessing models from multiple providers with one API key
- Comparing models across different providers
- Automatic fallback and load balancing
- Cost optimization across providers

**Setup:**

```bash
# Get your API key from https://openrouter.ai/keys
export OPENROUTER_API_KEY="sk-or-..."

# Use OpenRouter models (prefix with openrouter/)
python3 "$DEBATE_PY" critique --models openrouter/openai/gpt-4o,openrouter/anthropic/claude-3.5-sonnet < spec.md
```

**Popular OpenRouter models:**
- `openrouter/openai/gpt-4o` - GPT-4o via OpenRouter
- `openrouter/anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet
- `openrouter/google/gemini-2.0-flash` - Gemini 2.0 Flash
- `openrouter/meta-llama/llama-3.3-70b-instruct` - Llama 3.3 70B
- `openrouter/qwen/qwen-2.5-72b-instruct` - Qwen 2.5 72B

See the full model list at [openrouter.ai/models](https://openrouter.ai/models).

## Codex CLI Support

[Codex CLI](https://github.com/openai/codex) allows ChatGPT Pro subscribers to use OpenAI models without separate API credits. Models prefixed with `codex/` are routed through the Codex CLI.

**Setup:**

```bash
# Install Codex CLI (requires ChatGPT Pro subscription)
npm install -g @openai/codex

# Use Codex models (prefix with codex/)
python3 "$DEBATE_PY" critique --models codex/gpt-5.3-codex,gemini/gemini-2.0-flash < spec.md
```

**Reasoning effort:**

Control how much thinking time the model uses with `--codex-reasoning`:

```bash
# Available levels: low, medium, high, xhigh (default: xhigh)
python3 "$DEBATE_PY" critique --models codex/gpt-5.3-codex --codex-reasoning high < spec.md
```

Higher reasoning effort produces more thorough analysis but uses more tokens.

**Available Codex models:**
- `codex/gpt-5.3-codex` - GPT-5.3 via Codex CLI
- `codex/gpt-5.2-codex` - GPT-5.2 via Codex CLI

Check Codex CLI installation status:

```bash
python3 "$DEBATE_PY" providers
```

## Gemini CLI Support

[Gemini CLI](https://github.com/google-gemini/gemini-cli) allows Google account holders to use Gemini models without separate API credits. Models prefixed with `gemini-cli/` are routed through the Gemini CLI.

**Setup:**

```bash
# Install Gemini CLI
npm install -g @google/gemini-cli && gemini auth

# Use Gemini CLI models (prefix with gemini-cli/)
python3 "$DEBATE_PY" critique --models gemini-cli/gemini-3-pro-preview < spec.md
```

**Available Gemini CLI models:**
- `gemini-cli/gemini-3-pro-preview` - Gemini 3 Pro via CLI
- `gemini-cli/gemini-3-flash-preview` - Gemini 3 Flash via CLI

Check Gemini CLI installation status:

```bash
python3 "$DEBATE_PY" providers
```

## OpenAI-Compatible Endpoints

For models that expose an OpenAI-compatible API (local LLMs, self-hosted models, alternative providers), set `OPENAI_API_BASE`:

```bash
# Point to a custom endpoint
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://your-endpoint.com/v1"

# Use with any model name
python3 "$DEBATE_PY" critique --models gpt-4o < spec.md
```

This works with:
- Local LLM servers (Ollama, vLLM, text-generation-webui)
- OpenAI-compatible providers
- Self-hosted inference endpoints

## Usage

**Start from scratch (Codex):**

Ask Codex:
```
Use adversarial-spec to draft a PRD for a rate limiter service with a Redis backend.
```

**Refine an existing document (Codex):**

Ask Codex:
```
Use adversarial-spec on ./docs/my-spec.md
```

You will be prompted for:

1. **Document type**: PRD (business/product focus) or tech spec (engineering focus)
2. **Interview mode**: Optional in-depth requirements gathering session
3. **Opponent models**: Comma-separated list (e.g., `gpt-4o,gemini/gemini-2.0-flash,xai/grok-3`)

More models = more perspectives = stricter convergence.

## Document Types

### PRD (Product Requirements Document)

For stakeholders, PMs, and designers.

**Sections:** Executive Summary, Problem Statement, Target Users/Personas, User Stories, Functional Requirements, Non-Functional Requirements, Success Metrics, Scope (In/Out), Dependencies, Risks

**Critique focuses on:** Clear problem definition, well-defined personas, measurable success criteria, explicit scope boundaries, no technical implementation details

### Technical Specification

For developers and architects.

**Sections:** Overview, Goals/Non-Goals, System Architecture, Component Design, API Design (full schemas), Data Models, Infrastructure, Security, Error Handling, Performance/SLAs, Observability, Testing Strategy, Deployment Strategy

**Critique focuses on:** Complete API contracts, data model coverage, security threat mitigation, error handling, specific performance targets, no ambiguity for engineers

## Core Features

### Interview Mode

Before the debate begins, opt into an in-depth interview session to capture requirements upfront.

**Covers:** Problem context, users/stakeholders, functional requirements, technical constraints, UI/UX, tradeoffs, risks, success criteria

The interview uses probing follow-up questions and challenges assumptions. After completion, Codex synthesizes answers into a complete spec before starting the adversarial debate.

### Codex's Active Participation

Each round, Codex:

1. Reviews opponent critiques for validity
2. Provides independent critique (what did opponents miss?)
3. States agreement/disagreement with specific points
4. Synthesizes all feedback into revisions

Display format:

```
--- Round N ---
Opponent Models:
- [GPT-4o]: critiqued: missing rate limit config
- [Gemini]: agreed

Codex's Critique:
Security section lacks input validation strategy. Adding OWASP top 10 coverage.

Synthesis:
- Accepted from GPT-4o: rate limit configuration
- Added by Codex: input validation, OWASP coverage
- Rejected: none
```

### Early Agreement Verification

If a model agrees within the first 2 rounds, Codex is skeptical. The model is pressed to:

- Confirm it read the entire document
- List specific sections reviewed
- Explain why it agrees
- Identify any remaining concerns

This prevents false convergence from models that rubber-stamp without thorough review.

### User Review Period

After all models agree, you enter a review period with three options:

1. **Accept as-is**: Document is complete
2. **Request changes**: Codex updates the spec, you iterate without a full debate cycle
3. **Run another cycle**: Send the updated spec through another adversarial debate

### Additional Review Cycles

Run multiple cycles with different strategies:

- First cycle with fast models (gpt-4o), second with stronger models (o1)
- First cycle for structure/completeness, second for security focus
- Fresh perspective after user-requested changes

### PRD to Tech Spec Flow

When a PRD reaches consensus, you're offered the option to continue directly into a Technical Specification based on the PRD. This creates a complete documentation pair in a single session.

## Advanced Features

### Critique Focus Modes

Direct models to prioritize specific concerns:

```bash
--focus security      # Auth, input validation, encryption, vulnerabilities
--focus scalability   # Horizontal scaling, sharding, caching, capacity
--focus performance   # Latency targets, throughput, query optimization
--focus ux            # User journeys, error states, accessibility
--focus reliability   # Failure modes, circuit breakers, disaster recovery
--focus cost          # Infrastructure costs, resource efficiency
```

### Model Personas

Have models critique from specific professional perspectives:

```bash
--persona security-engineer      # Thinks like an attacker
--persona oncall-engineer        # Cares about debugging at 3am
--persona junior-developer       # Flags ambiguity and tribal knowledge
--persona qa-engineer            # Missing test scenarios
--persona site-reliability       # Deployment, monitoring, incidents
--persona product-manager        # User value, success metrics
--persona data-engineer          # Data models, ETL implications
--persona mobile-developer       # API design for mobile
--persona accessibility-specialist  # WCAG, screen readers
--persona legal-compliance       # GDPR, CCPA, regulatory
```

Custom personas also work: `--persona "fintech compliance officer"`

### Context Injection

Include existing documents for models to consider:

```bash
--context ./existing-api.md --context ./schema.sql
```

Use cases:
- Existing API documentation the new spec must integrate with
- Database schemas the spec must work with
- Design documents or prior specs for consistency
- Compliance requirements documents

`CONSTITUTION.md` in the current project root is auto-included for `critique` runs when present, so model assumptions stay aligned with project scope.

### Session Persistence and Resume

Long debates can crash or need to pause. Sessions save state automatically:

```bash
# Start a named session
echo "spec" | python3 "$DEBATE_PY" critique --models gpt-4o --session my-feature-spec

# Resume where you left off
python3 "$DEBATE_PY" critique --resume my-feature-spec

# List all sessions
python3 "$DEBATE_PY" sessions
```

Sessions save:
- Current spec state
- Round number
- All configuration (models, focus, persona, etc.)
- History of previous rounds

Sessions are stored in `~/.config/adversarial-spec/sessions/`.

### Auto-Checkpointing

When using sessions, each round's spec is saved to `.adversarial-spec-checkpoints/`:

```
.adversarial-spec-checkpoints/
├── my-feature-spec-round-1.md
├── my-feature-spec-round-2.md
└── my-feature-spec-round-3.md
```

Use these to rollback if a revision makes things worse.

### Preserve Intent Mode

Convergence can sand off novel ideas when models interpret "unusual" as "wrong". The `--preserve-intent` flag makes removal expensive:

```bash
--preserve-intent
```

When enabled, models must:

1. **Quote exactly** what they want to remove or substantially change
2. **Justify the harm** - not just "unnecessary" but what concrete problem it causes
3. **Distinguish error from preference** - only remove things that are factually wrong, contradictory, or risky
4. **Ask before removing** unusual but functional choices: "Was this intentional?"

This shifts the default from "sand off anything unusual" to "add protective detail while preserving distinctive choices."

Use when:
- Your spec contains intentional unconventional choices
- You want models to challenge your ideas, not homogenize them
- Previous rounds removed things you wanted to keep

### Cost Tracking

Every critique round displays token usage and estimated cost:

```
=== Cost Summary ===
Total tokens: 12,543 in / 3,221 out
Total cost: $0.0847

By model:
  gpt-4o: $0.0523 (8,234 in / 2,100 out)
  gemini/gemini-2.0-flash: $0.0324 (4,309 in / 1,121 out)
```

### Saved Profiles

Save frequently used configurations:

```bash
# Create a profile
python3 "$DEBATE_PY" save-profile strict-security \
  --models gpt-4o,gemini/gemini-2.0-flash \
  --focus security \
  --doc-type tech

# Use a profile
python3 "$DEBATE_PY" critique --profile strict-security < spec.md

# List profiles
python3 "$DEBATE_PY" profiles
```

Profiles are stored in `~/.config/adversarial-spec/profiles/`.

### Diff Between Rounds

See exactly what changed between spec versions:

```bash
python3 "$DEBATE_PY" diff --previous round1.md --current round2.md
```

### Export to Task List

Extract actionable tasks from a finalized spec:

```bash
cat spec-output.md | python3 "$DEBATE_PY" export-tasks --models gpt-4o --doc-type prd
```

Output includes title, type, priority, description, and acceptance criteria.

Use `--json` for structured output suitable for importing into issue trackers.

## Telegram Integration (Optional)

Get notified on your phone and inject feedback during the debate.

**Setup:**

```bash
export TELEGRAM_PY="$HOME/.codex/skills/adversarial-spec/scripts/telegram_bot.py"
# Or, if running from the repo:
# export TELEGRAM_PY="$PWD/skills/adversarial-spec/scripts/telegram_bot.py"
```

1. Message @BotFather on Telegram, send `/newbot`, follow prompts
2. Copy the bot token
3. Run: `python3 "$TELEGRAM_PY" setup`
4. Message your bot, run setup again to get your chat ID
5. Set environment variables:

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
```

**Features:**

- Async notifications when rounds complete (includes cost)
- 60-second window to reply with feedback (incorporated into next round)
- Final document sent to Telegram when debate concludes

## Output

Final document is:

- Complete, following full structure for document type
- Vetted by all models until unanimous agreement
- Ready for stakeholders without further editing

Output locations:

- Printed to terminal
- Written to `spec-output.md` (PRD) or `tech-spec-output.md` (tech spec)
- Sent to Telegram (if enabled)

Debate summary includes rounds completed, cycles run, models involved, Codex's contributions, cost, and key refinements made.

## CLI Reference

```bash
# Core commands
python3 "$DEBATE_PY" critique --models MODEL_LIST --doc-type TYPE [OPTIONS] < spec.md
python3 "$DEBATE_PY" critique --resume SESSION_ID
python3 "$DEBATE_PY" diff --previous OLD.md --current NEW.md
python3 "$DEBATE_PY" export-tasks --models MODEL --doc-type TYPE [--json] < spec.md

# Info commands
python3 "$DEBATE_PY" providers      # List providers and API key status
python3 "$DEBATE_PY" focus-areas    # List focus areas
python3 "$DEBATE_PY" personas       # List personas
python3 "$DEBATE_PY" profiles       # List saved profiles
python3 "$DEBATE_PY" sessions       # List saved sessions

# Profile management
python3 "$DEBATE_PY" save-profile NAME --models ... [--focus ...] [--persona ...]

# Bedrock management
python3 "$DEBATE_PY" bedrock status                      # Show Bedrock configuration
python3 "$DEBATE_PY" bedrock enable --region REGION      # Enable Bedrock mode
python3 "$DEBATE_PY" bedrock disable                     # Disable Bedrock mode
python3 "$DEBATE_PY" bedrock add-model MODEL             # Add model to available list
python3 "$DEBATE_PY" bedrock remove-model MODEL          # Remove model from list
python3 "$DEBATE_PY" bedrock list-models                 # List built-in model mappings
```

**Options:**
- `--models, -m` - Comma-separated model list (auto-detects from available API keys if not specified)
- `--doc-type, -d` - prd or tech
- `--codex-reasoning` - Reasoning effort for Codex models (low, medium, high, xhigh; default: xhigh)
- `--focus, -f` - Focus area (security, scalability, performance, ux, reliability, cost)
- `--persona` - Professional persona
- `--context, -c` - Context file (repeatable)
- `--profile` - Load saved profile
- `--preserve-intent` - Require justification for removals
- `--session, -s` - Session ID for persistence and checkpointing
- `--resume` - Resume a previous session
- `--press, -p` - Anti-laziness check
- `--telegram, -t` - Enable Telegram
- `--json, -j` - JSON output

## File Structure

```
adversarial-spec/
├── README.md
├── LICENSE
└── skills/
    └── adversarial-spec/
        ├── SKILL.md          # Skill definition and process
        └── scripts/
            ├── debate.py     # Multi-model debate orchestration
            └── telegram_bot.py   # Telegram notifications
```

## License

MIT
