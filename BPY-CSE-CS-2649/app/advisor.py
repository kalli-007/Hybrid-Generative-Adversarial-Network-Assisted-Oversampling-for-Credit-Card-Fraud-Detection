def risk_level(probability):
    if probability >= 0.85:
        return "Critical"
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


def ai_suggestions(probability, features=None):
    features = features or {}
    level = risk_level(probability)
    suggestions = []

    if level == "Critical":
        suggestions.extend([
            "Temporarily block the transaction and place the card under review.",
            "Contact the cardholder immediately using a verified channel.",
            "Require biometric or OTP verification before any retry.",
            "Review recent transactions from the same device, merchant and location.",
        ])
    elif level == "High":
        suggestions.extend([
            "Hold the transaction for step-up authentication.",
            "Verify device, merchant and location consistency.",
            "Monitor the account closely for the next 24 hours.",
        ])
    elif level == "Medium":
        suggestions.extend([
            "Request an additional OTP confirmation.",
            "Check whether the purchase matches the customer's normal behaviour.",
            "Continue enhanced monitoring for similar attempts.",
        ])
    else:
        suggestions.extend([
            "The transaction may be approved under normal controls.",
            "Continue routine account monitoring.",
        ])

    if float(features.get("ForeignTransaction", 0) or 0) == 1:
        suggestions.append("Confirm that the customer is currently making an international purchase.")
    if float(features.get("TransactionsLast24h", 0) or 0) >= 10:
        suggestions.append("Investigate the unusually high transaction frequency during the last 24 hours.")
    if float(features.get("Amount", 0) or 0) >= 50000:
        suggestions.append("Apply enhanced verification because the transaction amount is unusually high.")
    return level, suggestions


def batch_narrative(summary):
    total = summary["total"]
    fraud = summary["fraud"]
    rate = summary["fraud_rate"]
    critical = summary["risk_counts"].get("Critical", 0)
    high = summary["risk_counts"].get("High", 0)
    if fraud == 0:
        return (
            f"The file contains {total} transactions and no transaction crossed the model's fraud threshold. "
            "Routine monitoring is sufficient, although unusual high-value activity should still be reviewed."
        )
    return (
        f"The proposed model analyzed {total} transactions and flagged {fraud} ({rate:.2f}%) as potentially "
        f"fraudulent. {critical} are critical risk and {high} are high risk. Prioritize these transactions for "
        "cardholder verification, then review repeated activity, unusual locations and high-value payments."
    )

