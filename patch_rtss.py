import re
p = "Site_Setup.py"
s = open(p).read()

old = 'events += ["Bednet_Got_New_One", "Received_SMC"]'
assert s.count(old) == 1, "recorder line not found once"
s = s.replace(old, 'events += ["Bednet_Got_New_One", "Received_SMC", "Received_IRS"]')

anchor = "WRITE_PICKUP_STATE = True"
assert s.count(anchor) == 1, "WRITE_PICKUP_STATE not found once"
s = s.replace(anchor, anchor + '''

# RTS,S (EMOD RTSSVaccine). Values from the EMOD RTSSVaccine docs example;
# Boosted_Antibody_Concentration is a calibration choice, not derived from your data.
RTSS_BOOSTED_ANTIBODY = 1200
RTSS_COST_TO_CONSUMER = 1.0
RTSS_MAX_AGE_YEARS = 59 / 12.0''')

assert len(re.findall(r"def add_vaccine\(", s)) == 1
m = re.search(r"def add_vaccine\(.*?(?=\n\n\S)", s, re.S)
assert m, "could not find end of add_vaccine"
print("replacing add_vaccine:", m.group(0).count("\n") + 1, "lines")

new = '''def add_vaccine(campaign, cohorts):
    """RTS,S via the EMOD RTSSVaccine class (see EMOD docs).
    D1: Births -> wait dose-1 age -> D1. Dk: receipt of D(k-1) of the same
    birth cohort -> wait the age gap -> Dk, coverage = conditional uptake."""
    from emod_api import schema_to_class as s2c
    from emodpy_malaria.interventions.common import add_triggered_campaign_delay_event

    for c in cohorts:
        year = int(c["year"])
        first_day = (year - PICKUP_START_YEAR) * DAYS_PER_YEAR + 1
        for k, cond in enumerate(c["cond"], start=1):
            if cond <= 0:
                break
            iv = s2c.get_class_with_defaults("RTSSVaccine", campaign.schema_path)
            iv.Intervention_Name = f"RTSS_D{k}_{year}"
            iv.Boosted_Antibody_Concentration = RTSS_BOOSTED_ANTIBODY
            iv.Cost_To_Consumer = RTSS_COST_TO_CONSUMER
            announce = common.BroadcastEvent(campaign, Event_Trigger=f"Received_Vaccine_D{k}_{year}")
            if k == 1:
                trigger, start, listen = ["Births"], first_day, DAYS_PER_YEAR
                delay = c["age_days"][0]
            else:
                trigger = [f"Received_Vaccine_D{k - 1}_{year}"]
                start, listen = 1, -1
                delay = max(c["age_days"][k - 1] - c["age_days"][k - 2], 1)
            add_triggered_campaign_delay_event(
                campaign, start_day=int(start), trigger_condition_list=trigger,
                listening_duration=int(listen), delay_period_constant=float(delay),
                demographic_coverage=float(min(cond, 1.0)),
                target_age_min=0, target_age_max=RTSS_MAX_AGE_YEARS,
                individual_intervention=[iv, announce])'''
s = s[:m.start()] + new + s[m.end():]
open(p, "w").write(s)
print("patched")
