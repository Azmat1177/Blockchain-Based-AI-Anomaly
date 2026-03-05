# Blockchain-Based-AI-Anomaly
- https://convegni.unica.it/dlt2026/dlt-workshop/
- goal
- architecture
- https://iotascan.com/testnet/tx/8pWqQ3Jbj9So2eVNGvQY7Qrbst56ji7R3W1d87YZvwhs
- Tropic Square is an Company that produces cryptographic chips on the hardware side
- keywords: IOTA, AI, Anamoly-detection,

- Abstract:
  This paper introduces CheeseSafe, an IoT environmental monitoring system deployed in a commercial cheese storage facility in Sardinia, Italy. Building on our earlier work, PizzaSafe, which tracked temperature and humidity during pizza production with a threshold-based alert system on the AssetChain distributed ledger, CheeseSafe features three key upgrades: (1) the alert system is replaced by an unsupervised machine learning model called Isolation Forest, trained on real sensor data from a DHT22 device inside a cheese fridge, allowing it to detect statistical anomalies beyond simple thresholds; (2) the AssetChain ledger is replaced by the IOTA Tangle, a fee-free directed acyclic graph optimized for frequent IoT data anchoring; and (3) the TROPIC01 secure element is integrated for hardware cryptographic protection alongside software-level ECDSA signing, creating a dual-security framework. The target storage conditions are 1.6–6°C with 30–60% humidity. Analysis of 13 days of data from a real deployment between October 2025 and March 2026 shows the fridge operated at an average of 8.0°C — which exceeded the safe upper limit by 2–5°C for 96.1% of the time — highlighting the importance of automated anomaly detection in food safety cold chains. The Isolation Forest model achieved over 95% agreement with rule-based ground truth but also identified statistical deviations missed by simple thresholds. All anomalies were cryptographically signed by the cryptographic chip and securely anchored on the IOTA Tangle, ensuring an immutable audit trail in compliance with EU food safety regulations.
