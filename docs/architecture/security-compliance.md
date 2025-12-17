
# Security & Compliance
**Version:** 1.0.0
**Last Updated:** December 15, 2025
**Maintainer:** Ordinis Core Team

---

## Overview
This document covers authentication, authorization, and regulatory compliance.

## Security Measures
- API Key Management
- Environment Variables
- TLS 1.3 enforcement
- Secrets in vault, never in logs
- MFA required for production

## Compliance
- SEC/FINRA Considerations
- Audit Trails
- Governance policy enforcement (see `governance.yml`)

---

## Governance Reference
Security and compliance are governed by the policies in `governance.yml`, including:
- API security and key rotation
- Secrets management and encryption
- Audit logging and retention
- Regulatory reporting and review
