CREATE TABLE IF NOT EXISTS transactions (
    txn_id BIGINT PRIMARY KEY,
    user_id BIGINT,
    merchant_id BIGINT,
    amount DECIMAL(10,2),
    timestamp TIMESTAMP,
    channel VARCHAR(10),
    device_id VARCHAR(64),
    ip_country CHAR(2),
    merchant_category VARCHAR(50),
    distance_home FLOAT,
    previous_declines_24h INT,
    is_fraud TINYINT
);

CREATE INDEX idx_user_time ON transactions(user_id, timestamp);
CREATE INDEX idx_merchant_time ON transactions(merchant_id, timestamp);
