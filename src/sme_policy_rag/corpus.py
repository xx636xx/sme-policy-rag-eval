"""Synthetic SME Compliance, Cybersecurity (ISO 27001), and Workplace Safety (PRL) corpus & eval set."""

from __future__ import annotations

from .schemas import EvalCase, PolicyDocument


def load_policy_corpus() -> list[PolicyDocument]:
    """Return 8 structured Markdown operational manuals (24 distinct policy sections)."""
    return [
        PolicyDocument(
            doc_id="POL-SEC-01",
            title="ISO 27001 Access Control and Authentication Policy",
            domain="Cybersecurity",
            content_markdown="""# ISO 27001 Access Control and Authentication Policy

## Multi-Factor Authentication (ISO A.8.5)
All employee accounts accessing corporate email, VPN, ERP, or cloud consoles must enforce phishing-resistant Multi-Factor Authentication (MFA) under ISO 27001 control A.8.5. SMS-only one-time codes are prohibited for administrative accounts; IT Administrators must use FIDO2 hardware keys or app-based TOTP tokens.

## Password Policy and Lockout Thresholds
Minimum passphrase length is 14 characters for standard users and 18 characters for privileged accounts. Accounts automatically lock for 30 minutes after 5 consecutive failed authentication attempts. Password reuse is blocked for the previous 12 credentials, and compromised passwords are checked against breach dictionaries.

## Privileged Access Reviews (ISO A.8.2)
Privileged administrative access follows least-privilege principles and requires written approval from the IT Security Lead. All elevated roles must be reviewed quarterly (every 90 days), and unused administrative accounts are revoked automatically after 45 days of inactivity.
""",
        ),
        PolicyDocument(
            doc_id="POL-SEC-02",
            title="Cyber Incident Response and GDPR Breach Notification",
            domain="Cybersecurity",
            content_markdown="""# Cyber Incident Response and GDPR Breach Notification

## Severity Classification and Triage SLAs
Security incidents are categorized as P1 Critical, P2 High, or P3 Medium. P1 Critical incidents (ransomware, active data exfiltration, or compromised domain admin) require containment initiation within 30 minutes by the Incident Response Lead. P2 High incidents require initial triage within 4 hours.

## GDPR Personal Data Breach Reporting (72-Hour Rule)
Under GDPR Article 33 and ISO 27001 control A.5.24, any confirmed breach involving personal data must be notified to the Data Protection Authority (AEPD) within 72 hours of awareness by the Data Protection Officer (DPO). If the breach poses high risk to individuals, affected data subjects must be informed without undue delay.

## Forensic Evidence Preservation
Affected endpoints must be isolated from the network rather than powered off, preserving volatile RAM memory and authentication logs. Audit logs must be retained in immutable storage for a minimum of 12 months (365 days) under ISO 27001 control A.8.15.
""",
        ),
        PolicyDocument(
            doc_id="POL-SEC-03",
            title="Endpoint Encryption and Device Hardening Standard",
            domain="Cybersecurity",
            content_markdown="""# Endpoint Encryption and Device Hardening Standard

## Full-Disk Encryption with TPM 2.0 and BitLocker
Every corporate Windows laptop must enable BitLocker full-disk encryption (AES-256) backed by hardware TPM 2.0 before device handover. Recovery keys must be escrowed exclusively in the corporate MDM vault and never stored on USB drives or emailed in plaintext.

## Automatic Screen Lock and Patch Cadence
Workstations must enforce automatic screen locking after 10 minutes of idle time (5 minutes for finance and HR laptops). Critical operating system security patches (CVSS >= 9.0) must be deployed within 72 hours, and standard monthly patches within 14 days by IT Operations.

## Lost or Stolen Device Protocol
Employees must report a lost or stolen corporate laptop or smartphone to the IT Helpdesk within 2 hours of discovery. IT Operations immediately revokes active session tokens, rotates VPN certificates, and issues a remote wipe command.
""",
        ),
        PolicyDocument(
            doc_id="POL-OPS-04",
            title="Backup, Disaster Recovery, and Business Continuity",
            domain="Operations",
            content_markdown="""# Backup, Disaster Recovery, and Business Continuity

## 3-2-1-1-0 Backup Architecture
Critical databases and ERP repositories follow the 3-2-1-1-0 rule: 3 copies of data on 2 different media types, 1 offsite copy, 1 immutable air-gapped copy protected against ransomware deletion, and 0 unverified errors after automated restore checks.

## Recovery Time and Recovery Point Objectives (RTO / RPO)
Tier-1 production systems (billing, customer portal, SQL databases) have a Recovery Point Objective (RPO) of 1 hour and a Recovery Time Objective (RTO) of 4 hours. Tier-2 internal tools have an RPO of 24 hours and an RTO of 48 hours.

## Quarterly Restore Drills
IT Operations must execute and document full sandbox restore drills every 90 days (quarterly). Drill reports must be signed off by the Operations Director and audited annually under ISO 27001 control A.8.13.
""",
        ),
        PolicyDocument(
            doc_id="POL-AI-05",
            title="Generative AI Acceptable Use and Governance for SMEs",
            domain="Applied AI",
            content_markdown="""# Generative AI Acceptable Use and Governance for SMEs

## Prohibited Data Inputs on Public LLMs
Employees must never paste personal customer identifiers (DNI, phone numbers, health records), source-code secrets, API keys, or unreleased financial statements into consumer public LLM interfaces. Confidential document analysis must run exclusively on approved enterprise tenants with zero-data-retention agreements or on-premise local inference servers (such as Ollama).

## Human-in-the-Loop Verification Requirement
All AI-generated client deliverables, SQL queries, legal summaries, and analytical charts must undergo mandatory human review before external publication. Outputs containing numerical metrics must be cross-checked against primary source tables by the responsible analyst.

## Model Registry and ROI Tracking
Every departmental AI automation or assistant deployed in production must be logged in the AI Inventory Registry with its owner, data sources, risk tier under the EU AI Act, and measurable KPIs (hours saved per month, error rate, and quarterly ROI review).
""",
        ),
        PolicyDocument(
            doc_id="POL-PRL-06",
            title="Workplace Safety, Ergonomics, and Incident Investigation (PRL)",
            domain="Occupational Safety",
            content_markdown="""# Workplace Safety, Ergonomics, and Incident Investigation (PRL)

## Display Screen Equipment Ergonomics
For employees working with display screens more than 4 hours per day, monitors must be positioned at arm's length (50 to 70 cm) with the top edge at or slightly below eye level. Active micro-breaks of 5 minutes every 55 minutes of continuous screen work are mandatory to prevent musculoskeletal strain.

## Workplace Accident Notification and Root-Cause Analysis
Any workplace injury or near-miss must be logged by the supervisor within 24 hours in the PRL Safety Portal. Severe or lost-time accidents require formal root-cause investigation by the Occupational Safety Technician within 5 business days and statutory notification via the Delt@ system.

## Annual Medical Surveillance and Safety Training
All staff must receive job-specific occupational risk prevention training during onboarding (minimum 4 hours) and refresher sessions every 12 months. Voluntary periodic health checkups are scheduled annually through the accredited prevention service.
""",
        ),
        PolicyDocument(
            doc_id="POL-LAB-07",
            title="Chemical Laboratory Hygiene, PPE, and Spill Response",
            domain="Laboratory Safety",
            content_markdown="""# Chemical Laboratory Hygiene, PPE, and Spill Response

## Mandatory Laboratory Personal Protective Equipment (PPE)
Entry into wet-chemistry and instrumental analysis rooms requires safety goggles certified to EN 166, nitrile chemical-resistant gloves certified to EN 374, closed-toe footwear, and a flame-resistant cotton lab coat. Handling volatile acids or organic solvents must occur strictly inside certified fume hoods with face velocity between 0.4 and 0.6 m/s.

## Emergency Eye-Wash and Safety Shower Access
Emergency eye-wash stations and drench showers must remain unobstructed within 10 seconds walking distance (maximum 15 meters) from any corrosive-handling bench. Laboratory Technicians must perform and log weekly flushing checks every Monday to verify clean flow and tepid water temperature (16 to 38 degrees Celsius).

## Chemical Spill Containment and SDS Availability
Up-to-date Safety Data Sheets (SDS) must be accessible in both physical binders and the digital LIMS portal for 100% of stored reagents. Acid spills exceeding 250 mL require immediate area evacuation, neutralizer kit deployment by trained personnel, and notification to the Industrial Hygiene Coordinator within 15 minutes.
""",
        ),
        PolicyDocument(
            doc_id="POL-VND-08",
            title="Third-Party Vendor Security Assessment Policy",
            domain="Cybersecurity",
            content_markdown="""# Third-Party Vendor Security Assessment Policy

## Pre-Onboarding Security Due Diligence (ISO A.5.19)
Before onboarding any SaaS or cloud supplier that processes corporate or customer data, Procurement and IT Security must review valid ISO 27001 or SOC 2 Type II certification and execute a GDPR Data Processing Agreement (DPA).

## Supplier Risk Re-Assessment Schedule
Critical Tier-1 suppliers undergo formal security re-assessment every 12 months, including SLA compliance, sub-processor disclosures, and vulnerability remediation evidence. Contract termination clauses must mandate certified data deletion within 30 days of service offboarding.
""",
        ),
    ]


def load_eval_benchmark() -> list[EvalCase]:
    """Return 30 labeled RAG evaluation cases (24 answerable + 6 unanswerable abstention traps)."""
    return [
        # --- Exact Code / Standard Queries (8) ---
        EvalCase(
            case_id="Q01",
            query="What does ISO A.8.5 require for administrative accounts?",
            category="exact_code",
            expected_doc_id="POL-SEC-01",
            expected_section_keyword="Multi-Factor Authentication",
            required_answer_terms=["FIDO2", "TOTP", "SMS"],
        ),
        EvalCase(
            case_id="Q02",
            query="How frequently must privileged roles be reviewed under ISO A.8.2?",
            category="exact_code",
            expected_doc_id="POL-SEC-01",
            expected_section_keyword="Privileged Access Reviews",
            required_answer_terms=["90 days", "quarterly"],
        ),
        EvalCase(
            case_id="Q03",
            query="What is the audit log retention period required by ISO A.8.15?",
            category="exact_code",
            expected_doc_id="POL-SEC-02",
            expected_section_keyword="Forensic Evidence",
            required_answer_terms=["12 months", "365 days"],
        ),
        EvalCase(
            case_id="Q04",
            query="What hardware requirement and cipher are mandatory for BitLocker and TPM 2.0?",
            category="exact_code",
            expected_doc_id="POL-SEC-03",
            expected_section_keyword="Full-Disk Encryption",
            required_answer_terms=["AES-256", "TPM 2.0", "MDM"],
        ),
        EvalCase(
            case_id="Q05",
            query="Which certifications are required for laboratory goggles and gloves (EN 166 and EN 374)?",
            category="exact_code",
            expected_doc_id="POL-LAB-07",
            expected_section_keyword="Personal Protective Equipment",
            required_answer_terms=["EN 166", "EN 374", "nitrile"],
        ),
        EvalCase(
            case_id="Q06",
            query="What due diligence documents are needed before onboarding a SaaS vendor under ISO A.5.19?",
            category="exact_code",
            expected_doc_id="POL-VND-08",
            expected_section_keyword="Pre-Onboarding Security",
            required_answer_terms=["SOC 2", "DPA"],
        ),
        EvalCase(
            case_id="Q07",
            query="Under GDPR Article 33, who notifies the AEPD and within what deadline?",
            category="exact_code",
            expected_doc_id="POL-SEC-02",
            expected_section_keyword="GDPR Personal Data Breach",
            required_answer_terms=["72 hours", "DPO"],
        ),
        EvalCase(
            case_id="Q08",
            query="What does ISO A.8.13 require for restore drills?",
            category="exact_code",
            expected_doc_id="POL-OPS-04",
            expected_section_keyword="Quarterly Restore Drills",
            required_answer_terms=["90 days", "quarterly"],
        ),
        # --- Paraphrased / Semantic Policy Queries (8) ---
        EvalCase(
            case_id="Q09",
            query="Can staff upload client phone numbers or API keys into public ChatGPT?",
            category="paraphrased_policy",
            expected_doc_id="POL-AI-05",
            expected_section_keyword="Prohibited Data Inputs",
            required_answer_terms=["never", "Ollama"],
        ),
        EvalCase(
            case_id="Q10",
            query="Should an infected computer be turned off immediately during a cyberattack?",
            category="paraphrased_policy",
            expected_doc_id="POL-SEC-02",
            expected_section_keyword="Forensic Evidence",
            required_answer_terms=["isolated", "RAM"],
        ),
        EvalCase(
            case_id="Q11",
            query="Where should BitLocker recovery keys be saved instead of USB sticks?",
            category="paraphrased_policy",
            expected_doc_id="POL-SEC-03",
            expected_section_keyword="Full-Disk Encryption",
            required_answer_terms=["MDM", "USB"],
        ),
        EvalCase(
            case_id="Q12",
            query="What makes our backup strategy resilient against ransomware deleting backups?",
            category="paraphrased_policy",
            expected_doc_id="POL-OPS-04",
            expected_section_keyword="3-2-1-1-0 Backup",
            required_answer_terms=["immutable", "air-gapped"],
        ),
        EvalCase(
            case_id="Q13",
            query="Do analysts have to double-check AI-generated reports before sharing with clients?",
            category="paraphrased_policy",
            expected_doc_id="POL-AI-05",
            expected_section_keyword="Human-in-the-Loop",
            required_answer_terms=["human review", "primary source"],
        ),
        EvalCase(
            case_id="Q14",
            query="How far away should my office computer monitor sit and how often should I pause?",
            category="paraphrased_policy",
            expected_doc_id="POL-PRL-06",
            expected_section_keyword="Ergonomics",
            required_answer_terms=["50 to 70 cm", "5 minutes"],
        ),
        EvalCase(
            case_id="Q15",
            query="What happens if someone forgets their corporate laptop on a train?",
            category="paraphrased_policy",
            expected_doc_id="POL-SEC-03",
            expected_section_keyword="Lost or Stolen Device",
            required_answer_terms=["2 hours", "remote wipe"],
        ),
        EvalCase(
            case_id="Q16",
            query="What information needs to be recorded when launching a new company AI assistant?",
            category="paraphrased_policy",
            expected_doc_id="POL-AI-05",
            expected_section_keyword="Model Registry",
            required_answer_terms=["EU AI Act", "ROI"],
        ),
        # --- Procedural / Numeric Threshold Queries (8) ---
        EvalCase(
            case_id="Q17",
            query="After how many failed login attempts does an account lock and for how long?",
            category="procedural_threshold",
            expected_doc_id="POL-SEC-01",
            expected_section_keyword="Password Policy",
            required_answer_terms=["5", "30 minutes"],
        ),
        EvalCase(
            case_id="Q18",
            query="What is the containment SLA for P1 Critical ransomware incidents?",
            category="procedural_threshold",
            expected_doc_id="POL-SEC-02",
            expected_section_keyword="Severity Classification",
            required_answer_terms=["30 minutes", "P1"],
        ),
        EvalCase(
            case_id="Q19",
            query="What are the idle screen lock timeouts and critical CVSS patch deadlines?",
            category="procedural_threshold",
            expected_doc_id="POL-SEC-03",
            expected_section_keyword="Automatic Screen Lock",
            required_answer_terms=["10 minutes", "72 hours"],
        ),
        EvalCase(
            case_id="Q20",
            query="What are the RPO and RTO targets for Tier-1 production databases?",
            category="procedural_threshold",
            expected_doc_id="POL-OPS-04",
            expected_section_keyword="Recovery Time",
            required_answer_terms=["1 hour", "4 hours"],
        ),
        EvalCase(
            case_id="Q21",
            query="Within how many hours must a workplace near-miss be logged and severe accidents investigated?",
            category="procedural_threshold",
            expected_doc_id="POL-PRL-06",
            expected_section_keyword="Workplace Accident",
            required_answer_terms=["24 hours", "5 business days"],
        ),
        EvalCase(
            case_id="Q22",
            query="How close must emergency eye-wash stations be and when are weekly flush checks performed?",
            category="procedural_threshold",
            expected_doc_id="POL-LAB-07",
            expected_section_keyword="Eye-Wash",
            required_answer_terms=["10 seconds", "Monday"],
        ),
        EvalCase(
            case_id="Q23",
            query="What acid spill volume triggers immediate evacuation and who must be notified within 15 minutes?",
            category="procedural_threshold",
            expected_doc_id="POL-LAB-07",
            expected_section_keyword="Chemical Spill",
            required_answer_terms=["250 mL", "15 minutes"],
        ),
        EvalCase(
            case_id="Q24",
            query="How many days does a terminated supplier have to certify deletion of company data?",
            category="procedural_threshold",
            expected_doc_id="POL-VND-08",
            expected_section_keyword="Supplier Risk",
            required_answer_terms=["30 days"],
        ),
        # --- Unanswerable / Out-of-Scope Abstention Traps (6) ---
        EvalCase(
            case_id="U01",
            query="What is the maximum reimbursement limit for international business-class flights?",
            category="unanswerable",
            expected_doc_id=None,
            expected_section_keyword=None,
            answerable=False,
        ),
        EvalCase(
            case_id="U02",
            query="How many days of paid paternity leave does the company offer in Germany?",
            category="unanswerable",
            expected_doc_id=None,
            expected_section_keyword=None,
            answerable=False,
        ),
        EvalCase(
            case_id="U03",
            query="What is the dress code for attending external trade fairs and conferences?",
            category="unanswerable",
            expected_doc_id=None,
            expected_section_keyword=None,
            answerable=False,
        ),
        EvalCase(
            case_id="U04",
            query="How are employee stock options vested after a Series B funding round?",
            category="unanswerable",
            expected_doc_id=None,
            expected_section_keyword=None,
            answerable=False,
        ),
        EvalCase(
            case_id="U05",
            query="What catering budget is approved for quarterly team lunches in Madrid?",
            category="unanswerable",
            expected_doc_id=None,
            expected_section_keyword=None,
            answerable=False,
        ),
        EvalCase(
            case_id="U06",
            query="What mileage rate per kilometer is reimbursed for electric company fleet vans?",
            category="unanswerable",
            expected_doc_id=None,
            expected_section_keyword=None,
            answerable=False,
        ),
    ]
