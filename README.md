# End to End Network-anomaly-detection

## 🚨 Problem Statement

In the realm of cybersecurity, **network anomaly detection** is a critical task that involves identifying unusual patterns or behaviors that deviate from the norm within network traffic. These anomalies could signify a range of security threats, from compromised devices and malware infections to large-scale cyber-attacks like **DDoS (Distributed Denial of Service)**.

The challenge lies in accurately detecting these anomalies in real-time, amidst the vast and continuous streams of network data, which are often **noisy and heterogeneous**.

Traditional methods of network anomaly detection often rely on predefined rules or signatures based on known attack patterns. However, these methods fall short in detecting **new or evolving threats** that do not match the existing signatures. Furthermore, as network environments grow in complexity, maintaining and updating these rules becomes increasingly **cumbersome and less effective**.

---

## 📊 Dataset

The dataset used for this project is derived from the **KDD Cup 1999** benchmark dataset, which has been widely used for evaluating network intrusion detection systems.

### 🧾 Dataset Details:
- **Name:** KDD Cup 1999 
- **Source:** [UCI KDD Archive](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- **Instances:** ~5 million connection records (for KDD'99), ~125k (for NSL-KDD)
- **Features:** 41 features categorized into:
  - Basic connection features
  - Content features
  - Time-based traffic features
  - Host-based traffic features
- **Label classes:**
  - Normal (benign traffic)
  - Attack types grouped into: **DoS**, **Probe**, **R2L (Remote to Local)**, and **U2R (User to Root)**

This dataset provides both **labeled** and **unlabeled** network traffic, making it suitable for both **supervised** and **unsupervised** learning approaches.
