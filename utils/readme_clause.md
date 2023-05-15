### Arguments definition:

#### SELECT
* The DISTINCT keyword is an argument.
* Each of the comma-separated clauses is an argument, which could be a column name, a column calculation (col1 [+-*/] col2), or an aggregation (like sum(col1 + col2)).

#### FROM
* Each of the comma-separated clauses is an argument, which could be a table name or a SUBS clause (replaced subquery). The JOIN ON clause is ignored.

#### WHERE
* A condition which is separated by AND or OR is one argument. In the current implementation the difference between AND and OR is ignored.
* If a condition is nested with a subquery, the subquery is replaced with a speical SUBS clause and the entire condition is one argument.

#### GROUP BY
* Comma-separated clauses, which could be a column name or an aggregation.

#### HAVING
Same as WHERE.

#### ORDER BY
* The direction is one argument.
* Comma-separated clauses, which could be a column name or an aggregation.

#### LIMIT
* The value of the LIMIT number is one argument.

#### UNION / INTERSECT / EXCEPT
* SUBS is the only argument.

#### SUBS
recursively calculate the Edits between the two subqueries.

---