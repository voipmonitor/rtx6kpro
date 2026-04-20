# Outlier Report for Luke

This note explains why the main comparison should **exclude** the pathological breakdown runs from the headline statistics.

## Recommendation

Use the following filtered comparison as the main result:

```text
+-----------+-----------------------+-----------+---------+-------+-------------------------------+--------------------------------------+
| Variant   | Excluded runs         | Completed | Correct | Wrong | Completion tokens min/med/avg/max    | Elapsed s min/med/avg/max            |
+-----------+-----------------------+-----------+---------+-------+-------------------------------+--------------------------------------+
| dense_mla | 8                     | 29        | 22      | 7     | 3219 / 8631.0 / 8560.000 / 16875     | 36.813 / 100.648 / 103.488 / 189.575 |
| nsa       | 21, 29                | 28        | 25      | 3     | 1784 / 4965.5 / 5618.464 / 27297     | 25.270 / 76.562 / 97.013 / 615.137   |
+-----------+-----------------------+-----------+---------+-------+-------------------------------+--------------------------------------+
```

## Why exclude these runs?

Because these are not normal “the model reasoned badly” cases. They are pathological generation failures:

- `dense_mla` run `8`
  - truncated / looping reasoning
  - no clean final answer
- `nsa` run `21`
  - repetition / degeneration
  - `finish_reason=length`
  - `completion_tokens=40000`
- `nsa` run `29`
  - repetition / degeneration
  - `finish_reason=length`
  - `completion_tokens=40000`

For the NSA case in particular, runs `21` and `29` are exactly the kind of repetitive runaway bug that should not be allowed to dominate the top-line quality statistics.

## Why this matters

If those runaway NSA runs are left in the main table:

- average completion tokens inflate sharply
- average elapsed time becomes misleading
- mean throughput becomes dominated by a bug tail instead of normal behavior

That is useful as **stability evidence**, but it is not the right top-line metric if the goal is to compare the actual dense-vs-NSA computation quality on successful/coherent runs.

## Correct interpretation

There are two separate questions:

1. Which path is more faithful / accurate on the task?
2. Which path has worse catastrophic tail failures?

The filtered statistics should answer question 1.
The raw all-run statistics should answer question 2.

## Bottom line

- For **quality comparison**, use the filtered statistics above.
- For **stability / tail-risk discussion**, keep the raw outlier runs in the appendix and full-output files.

That is the cleanest way to present the result to Luke without hiding the bug, while also not letting the bug corrupt the main comparison.
