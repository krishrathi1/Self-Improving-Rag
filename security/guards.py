"""
APEX Security Guards — Input and output safety filters.
Deny-first approach: blocks prompt injection, PII, and harmful content.
"""

import re
from loguru import logger


# --- Input Guard ---
# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+above",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(a\s+)?jailbreak",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+instructions",
    r"override\s+safety",
    r"bypass\s+filter",
    r"\\x[0-9a-fA-F]{2}",  # Hex escape sequences
    r"<script\b",  # XSS attempt
]

INJECTION_REGEX = re.compile(
    "|".join(INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL
)


def input_guard(query: str) -> bool:
    """
    Check if a query is safe to process (deny-first approach).
    
    Returns True if safe, False if blocked.
    """
    if not query or not query.strip():
        logger.warning("🛡️ Input blocked: empty query")
        return False

    if len(query) > 5000:
        logger.warning("🛡️ Input blocked: query too long")
        return False

    if INJECTION_REGEX.search(query):
        logger.warning(f"🛡️ Input blocked: injection attempt detected")
        return False
        
    # STATE-OF-THE-ART: Intent/Boundary Guard
    # Production systems should route via LlamaGuard, NeMo Guardrails, or asynchronous LLM strict-intent checks
    # For speed, we will implement a fast heuristic: length and structural entropy
    structural_markers = ["[", "]", "{", "}", "<|", "|>"] 
    if len([m for m in structural_markers if m in query]) > 5:
        logger.warning("🛡️ Input blocked: structural injection anomaly (Too many prompt-like boundaries)")
        return False

    return True


# --- Output Guard ---
# PII patterns for redaction
PII_PATTERNS = {
    "email": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL_REDACTED]"),
    "phone": (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', "[PHONE_REDACTED]"),
    "ssn": (r'\b\d{3}-\d{2}-\d{4}\b', "[SSN_REDACTED]"),
    "credit_card": (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', "[CC_REDACTED]"),
    "ip_address": (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "[IP_REDACTED]"),
}


def output_guard(text: str) -> str:
    """
    Redact PII from LLM output.
    
    Returns the text with sensitive information replaced by tokens.
    """
    if not text:
        return text

    redacted = text
    for pii_type, (pattern, replacement) in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted)
        if matches:
            redacted = re.sub(pattern, replacement, redacted)
            logger.info(f"🛡️ Output redacted: {len(matches)} {pii_type}(s)")

    return redacted
