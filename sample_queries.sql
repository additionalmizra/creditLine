-- Recent high‑risk patterns by merchant
SELECT merchant_id,
       AVG(amount) AS avg_amt,
       SUM(is_fraud) AS fraud_cnt,
       COUNT(*) AS txn_cnt,
       SUM(is_fraud)/COUNT(*) AS fraud_rate
FROM transactions
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY merchant_id
ORDER BY fraud_rate DESC
LIMIT 25;
