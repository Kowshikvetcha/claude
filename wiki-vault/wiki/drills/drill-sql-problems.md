---
title: SQL Problem Drill — 15 Problems, Easy to Hard
type: drill
domain: sql
roles: [data-scientist, ml-engineer, ai-engineer, mlops-engineer, fde]
difficulty: core
frequency: high
status: drafted
tags: [drill, sql]
updated: 2026-09-13
---

# SQL Problem Drill — 15 Problems, Easy to Hard

Work top to bottom without a scratchpad crutch — write the query, then check. Each problem gives a
tiny inline table so you can trace it by hand. See [[sql-fundamentals]], [[joins-deep-dive]],
[[window-functions]], [[aggregations-and-grouping]], [[subqueries-and-ctes]],
[[sql-analytics-patterns]] for the underlying concepts.

## 1. Second-highest salary (easy)

```text
employees(id, name, salary)
1, Asha, 90000
2, Ravi, 120000
3, Meera, 120000
4, Karan, 75000
```

**Q.** Find the second-highest *distinct* salary.

```sql
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

## 2. Duplicate rows (easy)

```text
signups(email, source)
a@x.com, ads
a@x.com, ads
b@x.com, organic
```

**Q.** Return rows that appear more than once (full-row duplicates).

```sql
SELECT email, source, COUNT(*) AS cnt
FROM signups
GROUP BY email, source
HAVING COUNT(*) > 1;
```

## 3. Delete duplicates, keep one (easy–medium)

```text
users(id, email)
1, a@x.com
2, a@x.com
3, b@x.com
```

**Q.** Keep the lowest `id` per email, delete the rest.

```sql
DELETE FROM users
WHERE id NOT IN (
  SELECT MIN(id) FROM users GROUP BY email
);
```

Row-number variant (works when there's no reliable `MIN(id)` tiebreak, e.g. dedup on a wider key):

```sql
DELETE FROM users u
USING (
  SELECT id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
  FROM users
) ranked
WHERE u.id = ranked.id AND ranked.rn > 1;
```

## 4. Top-N per group (medium)

```text
sales(rep, region, amount)
Asha, North, 500
Asha, North, 900
Ravi, North, 700
Meera, South, 1000
Meera, South, 400
```

**Q.** Top 1 sale by amount per region.

```sql
SELECT region, rep, amount
FROM (
  SELECT region, rep, amount,
         RANK() OVER (PARTITION BY region ORDER BY amount DESC) AS rnk
  FROM sales
) t
WHERE rnk = 1;
```

`RANK()` ties on equal amounts (both rows returned); use `ROW_NUMBER()` if you need exactly one row
per group even on a tie. See [[window-functions]] for the `RANK` vs `DENSE_RANK` vs `ROW_NUMBER`
distinction.

## 5. Running total (medium)

```text
orders(order_date, amount)
2026-01-01, 100
2026-01-02, 150
2026-01-03, 50
```

**Q.** Cumulative revenue by date.

```sql
SELECT order_date, amount,
       SUM(amount) OVER (ORDER BY order_date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM orders;
```

## 6. Moving average (medium)

```text
metrics(day, value)  -- 30 rows of daily values
```

**Q.** 7-day trailing moving average of `value`.

```sql
SELECT day, value,
       AVG(value) OVER (ORDER BY day
                         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d
FROM metrics;
```

## 7. Self-join: employees earning more than their manager (medium)

```text
employees(id, name, salary, manager_id)
1, Asha, 90000, NULL
2, Ravi, 120000, 1
3, Meera, 85000, 1
```

**Q.** List employees who earn more than their manager.

```sql
SELECT e.name AS employee, m.name AS manager, e.salary, m.salary AS manager_salary
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;
```

## 8. Self-join: consecutive login streak (medium–hard)

```text
logins(user_id, login_date)
1, 2026-01-01
1, 2026-01-02
1, 2026-01-03
1, 2026-01-05
2, 2026-01-01
```

**Q.** Find each user's longest run of *consecutive* daily logins — the classic
**gaps-and-islands** pattern.

```sql
WITH grouped AS (
  SELECT user_id, login_date,
         login_date - (ROW_NUMBER() OVER (
           PARTITION BY user_id ORDER BY login_date
         ))::int AS grp
  FROM logins
)
SELECT user_id, MIN(login_date) AS streak_start, MAX(login_date) AS streak_end,
       COUNT(*) AS streak_len
FROM grouped
GROUP BY user_id, grp
ORDER BY user_id, streak_len DESC;
```

The trick: subtracting a row's sequential rank from its date collapses every run of consecutive
dates onto the same constant (`grp`), turning "find consecutive runs" into a plain `GROUP BY`.

## 9. Gaps in a sequence (medium–hard)

```text
sensor_readings(id, reading_at)
1, 2026-01-01 00:00
2, 2026-01-01 00:01
3, 2026-01-01 00:04
```

**Q.** Find gaps of more than 1 minute between consecutive readings.

```sql
SELECT reading_at AS gap_start, next_reading AS gap_end,
       next_reading - reading_at AS gap_length
FROM (
  SELECT reading_at, LEAD(reading_at) OVER (ORDER BY reading_at) AS next_reading
  FROM sensor_readings
) t
WHERE next_reading - reading_at > INTERVAL '1 minute';
```

## 10. Cohort retention (hard)

```text
activity(user_id, signup_month, activity_month)
1, 2026-01, 2026-01
1, 2026-01, 2026-02
2, 2026-01, 2026-01
3, 2026-02, 2026-02
3, 2026-02, 2026-03
```

**Q.** Month-0/month-1/month-2 retention by signup cohort.

```sql
WITH cohort AS (
  SELECT user_id, signup_month,
         DATE_PART('month', AGE(activity_month, signup_month)) AS month_number
  FROM activity
),
cohort_size AS (
  SELECT signup_month, COUNT(DISTINCT user_id) AS cohort_users
  FROM cohort WHERE month_number = 0
  GROUP BY signup_month
)
SELECT c.signup_month, c.month_number,
       COUNT(DISTINCT c.user_id) AS active_users,
       ROUND(COUNT(DISTINCT c.user_id)::numeric / s.cohort_users, 3) AS retention_rate
FROM cohort c
JOIN cohort_size s USING (signup_month)
GROUP BY c.signup_month, c.month_number, s.cohort_users
ORDER BY c.signup_month, c.month_number;
```

See [[sql-analytics-patterns]] for the general cohort-table template.

## 11. Rolling N-day active users (hard)

```text
events(user_id, event_date)
```

**Q.** For each date, count *distinct* users active in the trailing 7 days (a true windowed
distinct count — plain `SUM() OVER` won't dedup users).

```sql
SELECT e1.event_date,
       COUNT(DISTINCT e2.user_id) AS dau_7d_trailing
FROM (SELECT DISTINCT event_date FROM events) e1
JOIN events e2
  ON e2.event_date BETWEEN e1.event_date - INTERVAL '6 days' AND e1.event_date
GROUP BY e1.event_date
ORDER BY e1.event_date;
```

This is an $O(n^2)$-shaped join on dates — fine for a small date range; at scale, precompute a
per-user first-seen date and use `COUNT(DISTINCT)` window emulation via `HyperLogLog`/approx-distinct
functions, or push it to Spark. See [[sql-query-optimization]].

## 12. First purchase attribution (medium–hard)

```text
orders(user_id, order_id, order_date, channel)
1, 101, 2026-01-01, ads
1, 102, 2026-01-10, organic
2, 201, 2026-01-05, referral
```

**Q.** Attribute each user's *first* order to a channel, then count revenue by that channel across
all their subsequent orders (first-touch attribution).

```sql
WITH first_touch AS (
  SELECT user_id, channel,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) AS rn
  FROM orders
)
SELECT ft.channel, COUNT(o.order_id) AS orders_attributed, SUM(o.amount) AS revenue
FROM first_touch ft
JOIN orders o ON o.user_id = ft.user_id
WHERE ft.rn = 1
GROUP BY ft.channel;
```

## 13. Median without a MEDIAN function (hard)

```text
scores(student_id, score)
```

**Q.** Compute the median score using only window functions (portable across engines lacking
`PERCENTILE_CONT`).

```sql
WITH ranked AS (
  SELECT score,
         ROW_NUMBER() OVER (ORDER BY score) AS rn,
         COUNT(*) OVER () AS total
  FROM scores
)
SELECT AVG(score) AS median
FROM ranked
WHERE rn IN (FLOOR((total + 1) / 2.0), CEIL((total + 1) / 2.0));
```

Where available, `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score)` is the direct one-liner —
know both; interviewers sometimes explicitly disallow the built-in.

## 14. Pivot without PIVOT (hard)

```text
survey(user_id, question, answer)
1, q1, yes
1, q2, no
2, q1, no
2, q2, yes
```

**Q.** Turn long-format survey answers into one row per user, one column per question.

```sql
SELECT user_id,
       MAX(CASE WHEN question = 'q1' THEN answer END) AS q1,
       MAX(CASE WHEN question = 'q2' THEN answer END) AS q2
FROM survey
GROUP BY user_id;
```

`MAX` here is just a way to collapse a single non-null value per group — it works because exactly
one row per `(user_id, question)` is non-null after the `CASE`.

## 15. Sessionization (hard)

```text
clicks(user_id, click_time)
```

**Q.** Group clicks into sessions where a new session starts after 30 minutes of inactivity;
return session start, end, and click count.

```sql
WITH marked AS (
  SELECT user_id, click_time,
         CASE WHEN click_time - LAG(click_time) OVER (
                PARTITION BY user_id ORDER BY click_time
              ) > INTERVAL '30 minutes'
              OR LAG(click_time) OVER (PARTITION BY user_id ORDER BY click_time) IS NULL
         THEN 1 ELSE 0 END AS is_new_session
  FROM clicks
),
sessioned AS (
  SELECT user_id, click_time,
         SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY click_time) AS session_id
  FROM marked
)
SELECT user_id, session_id, MIN(click_time) AS session_start,
       MAX(click_time) AS session_end, COUNT(*) AS clicks
FROM sessioned
GROUP BY user_id, session_id
ORDER BY user_id, session_id;
```

Same gap-collapsing idea as problem 8: a running `SUM` over a 0/1 "new session" flag produces a
monotonically increasing session id per user — a general technique worth recognising, not
memorising.

## Related

[[sql-fundamentals]] · [[joins-deep-dive]] · [[window-functions]] · [[aggregations-and-grouping]] ·
[[subqueries-and-ctes]] · [[sql-query-optimization]] · [[sql-analytics-patterns]] · [[qbank-sql]]
