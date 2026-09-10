# Service address rules — reviewed 6 September 2026

These sources establish provider addresses and authentication routes. They do not validate this project's rule as a phishing detector and do not prove that every page on an official service is safe. The extension uses narrower exact host/path rules than the providers' network allowlists.

- Tranco frozen list XN6XN: https://tranco-list.eu/list/XN6XN — generated 5 September 2026 from source rankings covering 7 August–5 September 2026. Its first 250 pay-level domains are stored for candidate selection and rank metadata only.
- Tranco methodology: https://tranco-list.eu/methodology — documents its aggregation of several popularity rankings and the limitations of popularity lists.
- Public Suffix List: https://publicsuffix.org/ — used to identify the registered domain for rank metadata; it is not a trust indicator.

- OpenAI: https://help.openai.com/en/articles/9247338-network-recommendations-for-chatgpt-errors-on-web-and-apps — identifies ChatGPT/OpenAI authentication hosts, including auth0.openai.com and chat.openai.com. Shared content and arbitrary subdomains are excluded from our rule.
- Google: https://developers.google.com/identity/openid-connect/reference — documents accounts.google.com/o/oauth2/v2/auth and its normal query parameters.
- Google: https://developers.google.com/identity/protocols/oauth2/web-server — documents the authorisation flow and redirect URI handling. The extension scans the eventual destination separately; it does not certify an OAuth application.
- Google Drive Help: https://support.google.com/drive/answer/2424384 — directs users to `drive.google.com` and identifies Docs, Sheets, Slides and Forms as Drive file types.
- Google Workspace Learning Center: https://support.google.com/a/users/answer/9282664 — identifies `docs.google.com` as the Google Docs web address.
- Google Docs Editors Help: https://support.google.com/docs/answer/49114 — covers Google documents, spreadsheets, presentations and forms and warns that hosted content can still be suspected of phishing or malware. The catalogue therefore identifies the service origin only; it does not certify hosted content.
- Microsoft: https://learn.microsoft.com/en-us/graph/auth-v2-user — documents login.microsoftonline.com authentication endpoints.

Local policy decisions: reviewed fixed first-party pages use quiet presentation. User-created Google Workspace content may receive a softened review at Caution, but High Risk and Very High Risk remain blocking in Normal mode. Redirect endpoints and unlisted hosts retain ordinary model treatment. These are explicit browser-policy decisions, not additional model training. Keep raw model metrics separate from results after applying these rules.
