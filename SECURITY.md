# Security Policy

This document outlines the security policy for the o3 Reasoning Framework project, including the associated Streamlit application and result analysis tool.

## Supported Versions

| Version | Supported          | Receiving Security Updates |
| ------- | ------------------ | -------------------------- |
| Current | :white_check_mark: | :white_check_mark:         |
| Older   | :x:                | :x:                        |

Only the current version of the o3 Reasoning Framework is actively supported and receives security updates.    Users are strongly encouraged to use the latest version to ensure they have the most secure experience.

## Reporting a Vulnerability

We take security vulnerabilities seriously.    If you discover a security vulnerability in the o3 Reasoning Framework, please report it to us immediately.

**Do not publicly disclose the vulnerability before we have had a chance to address it.**

To report a vulnerability, please follow these steps:

1. **Email:** Send an email to yuka@lumenlab.cc
2. **Subject:** Use the subject line "o3 Reasoning Framework Security Vulnerability Report".
3. **Details:** In your email, please provide the following information:
*   A clear description of the vulnerability.
*   Steps to reproduce the vulnerability.
*   The affected component(s) and version(s) of the o3 Reasoning Framework.
*   Any potential impact of the vulnerability.
*   Your contact information (name, email address).

We will acknowledge receipt of your report within 2 business days and will work to investigate and address the vulnerability as quickly as possible.

## Vulnerability Handling Process

1. **Triage:** We will review the vulnerability report and assess its severity and impact.
2. **Investigation:** We will investigate the vulnerability and determine the root cause.
3. **Fix:** We will develop and test a fix for the vulnerability.
4. **Release:** We will release a new version of the o3 Reasoning Framework that includes the fix.
5. **Disclosure:** We will publicly disclose the vulnerability and the fix, including any relevant details and mitigation steps.     We will credit the reporter of the vulnerability (unless they request anonymity).

## Security Considerations

The following security considerations should be taken into account when using the o3 Reasoning Framework:

*   **API Keys:** The o3 Reasoning Framework may require API keys for external services (e.g., OpenAI).    Store these API keys securely and do not hardcode them directly in your code.    Use environment variables or a secure configuration management system.
*   **Input Validation:** The Streamlit application accepts user input.    Implement proper input validation to prevent injection attacks (e.g., prompt injection).     Sanitize and validate all user-provided data.
*   **Dependency Management:** Keep all dependencies up to date to ensure you have the latest security patches.    Regularly review and update your `requirements.txt` file.
*   **File Storage:** If you enable result saving, ensure that the result directory is properly secured to prevent unauthorized access to sensitive data.     Consider encrypting the result files.
*   **Model Security:**  Be aware of potential vulnerabilities in the underlying language models (e.g., model poisoning, adversarial attacks).     Consider using model security tools and techniques to mitigate these risks.
*   **TDA Library (Gudhi):**  Ensure the Gudhi library is from a trusted source and kept up to date, as vulnerabilities in scientific computing libraries can be exploited.
*   **Authentication and Authorization (Streamlit):**  For production deployments of the Streamlit app, implement authentication and authorization to restrict access to authorized users only.    Streamlit provides built-in mechanisms for this.
*   **Code Review:** Regularly review the codebase for potential security vulnerabilities.
*   **Regular Security Audits:** Consider conducting regular security audits of the o3 Reasoning Framework to identify and address any potential security risks.

## Disclaimer

This security policy is subject to change without notice.    We are not responsible for any damages or losses resulting from security vulnerabilities in the o3 Reasoning Framework.     Users are responsible for taking appropriate security measures to protect their systems and data.
