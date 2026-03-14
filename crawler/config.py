"""
MOM target URLs and crawler configuration.
Add or remove URLs from MOM_URLS to control what gets scraped.
"""

MOM_URLS = [
    # Employment Act — specific content pages
    "https://www.mom.gov.sg/employment-practices/employment-act/who-is-covered",
    "https://www.mom.gov.sg/employment-practices/employment-act/contracts-of-service",
    # Leave & holidays
    "https://www.mom.gov.sg/employment-practices/leave-and-holidays/annual-leave",
    "https://www.mom.gov.sg/employment-practices/leave-and-holidays/sick-leave",
    "https://www.mom.gov.sg/employment-practices/leave-and-holidays/maternity-leave",
    "https://www.mom.gov.sg/employment-practices/leave-and-holidays/paternity-leave",
    "https://www.mom.gov.sg/employment-practices/leave-and-holidays/childcare-leave",
    "https://www.mom.gov.sg/employment-practices/leave-and-holidays/public-holidays",
    # Salary
    "https://www.mom.gov.sg/employment-practices/salary/paying-salary",
    "https://www.mom.gov.sg/employment-practices/salary/deducting-salary",
    # Termination
    "https://www.mom.gov.sg/employment-practices/termination-of-employment/termination-with-notice",
    "https://www.mom.gov.sg/employment-practices/termination-of-employment/termination-without-notice",
    "https://www.mom.gov.sg/employment-practices/termination-of-employment/wrongful-dismissal",
    # Employment Pass
    "https://www.mom.gov.sg/passes-and-permits/employment-pass/eligibility",
    "https://www.mom.gov.sg/passes-and-permits/employment-pass/apply-for-a-pass",
    "https://www.mom.gov.sg/passes-and-permits/employment-pass/renew-a-pass",
    "https://www.mom.gov.sg/passes-and-permits/employment-pass/cancel-a-pass",
    # S Pass
    "https://www.mom.gov.sg/passes-and-permits/s-pass/eligibility",
    "https://www.mom.gov.sg/passes-and-permits/s-pass/quota-and-levy",
    # Work Permit (foreign worker)
    "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/eligibility",
    "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-worker/quota-and-levy",
    # Work Permit (domestic worker)
    "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-domestic-worker/eligibility",
    "https://www.mom.gov.sg/passes-and-permits/work-permit-for-foreign-domestic-worker/salary-and-leave",
    # Work injury compensation
    "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/who-is-covered",
    "https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation/making-a-claim",
]

# Output directory for scraped markdown files (relative to project root)
OUTPUT_DIR = "../data/raw_markdown"

