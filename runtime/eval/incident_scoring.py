"""Preregistered incident scenario truth and contamination/recovery diagnostics."""

from collections import Counter, defaultdict


def member_label(row):
    return row["anomaly_type"] if row.get("incident_id") else "background"


def group_truth(rows, groups):
    truth = []
    for group in groups:
        if not group:
            raise ValueError("Incident groups cannot be empty")
        counts = Counter(member_label(rows[i]) for i in group)
        label, count = counts.most_common(1)[0]
        truth.append(label if count * 2 > len(group) else "mixed")
    return truth


def diagnostics(rows, groups):
    members = [i for group in groups for i in group]
    if not rows or sorted(members) != list(range(len(rows))):
        raise ValueError(
            "Predicted groups must partition every panel record exactly once"
        )
    incident_groups = defaultdict(set)
    covered_ids = set()
    covered_members = 0
    group_results = []
    merged = 0
    weighted_purity = 0
    for number, group in enumerate(groups):
        counts = Counter(member_label(rows[i]) for i in group)
        normal = sum(not rows[i].get("incident_id") for i in group)
        ids = {rows[i]["incident_id"] for i in group if rows[i].get("incident_id")}
        for identifier in ids:
            incident_groups[identifier].add(number)
        cross_source = len(group) >= 2 and len({rows[i]["source"] for i in group}) >= 2
        if cross_source:
            covered_ids.update(ids)
            covered_members += len(group) - normal
        merged += len(ids) > 1
        dominant = max(counts.values())
        weighted_purity += dominant
        group_results.append(
            dict(
                records=len(group),
                normal_members=normal,
                normal_member_fraction=normal / len(group),
                purity=dominant / len(group),
                distinct_incidents=len(ids),
                cross_source=cross_source,
            )
        )
    incident_members = sum(bool(r.get("incident_id")) for r in rows)
    return dict(
        groups=group_results,
        normal_member_fraction=1 - incident_members / len(rows),
        mean_group_purity=sum(g["purity"] for g in group_results) / len(group_results),
        member_weighted_purity=weighted_purity / len(rows),
        covered_incidents=len(covered_ids),
        total_incidents=len(incident_groups),
        incident_coverage=len(covered_ids) / len(incident_groups)
        if incident_groups
        else None,
        covered_incident_members=covered_members,
        total_incident_members=incident_members,
        incident_member_coverage=covered_members / incident_members
        if incident_members
        else None,
        merged_incident_groups=int(merged),
        split_incidents=sum(len(g) > 1 for g in incident_groups.values()),
    )
