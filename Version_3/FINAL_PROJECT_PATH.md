# Final demonstration path

## Selected implementation

Use **Phishing URL Guard v0.6.0** from:

`extension/`

This is the only active extension copy kept in the presentation workspace. It uses the frozen three-seed HYB-MS ensemble trained with 10% filtered MS-HardGAN-C augmentation. The model and decision thresholds remain frozen; v0.6.0 adds reliability fixes, bilingual English/Arabic pages, clearer controls, local consent-based feedback, trusted-service rules, and URL-signal explanations.

## Install for the demonstration

1. Disable any older copy of Phishing URL Guard.
2. Open `chrome://extensions` and enable **Developer mode**.
3. Select **Load unpacked**.
4. Choose the repository's `extension/` directory.
5. Confirm that the displayed version is `0.6.0`.
6. Open the extension popup and choose Normal or Passive mode.
7. Test only known safe pages and offline evaluation examples. Do not visit live phishing pages.

## What to demonstrate

- A normal legitimate service page.
- A long legitimate login or service URL.
- A safe, locally stored phishing example.
- The Caution advisory and the High Risk/Very High Risk interruption behavior.
- English and Arabic interfaces.
- The URL-signal explanation in the extension interface.
- Consent-based local feedback, repeated-URL deduplication, and JSON export.

In **Normal mode**, Caution is advisory. High Risk and Very High Risk interrupt navigation. In **Passive mode**, the extension reports its assessment without redirecting the page.

## Main research result

The controlled experiments show that limited, filtered GAN augmentation can improve the hybrid detector under some frozen internal and external conditions. The effect is conditional on the dataset and evaluation setting; it is not a claim that GAN augmentation improves every classifier or every metric.

## Scope of the claim

The extension is a graduation-project prototype suitable for an academic demonstration and supervised user evaluation. It is not a production-certified firewall and does not claim to predict all future or zero-day phishing attacks. Distribution shift, false warnings, missed phishing URLs, and asynchronous browser navigation remain documented limitations.
