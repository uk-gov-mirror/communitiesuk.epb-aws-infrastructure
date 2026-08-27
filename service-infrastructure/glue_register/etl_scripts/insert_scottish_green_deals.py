import sys
import pg8000
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions

args = getResolvedOptions(
    sys.argv,
    [
        "JOB_NAME",
        "INPUT_S3_PATH_GDPlans",
        "INPUT_S3_PATH_GD_P",
        "INPUT_S3_PATH_GDPDC",
        "INPUT_S3_PATH_GDPM",
        "INPUT_S3_PATH_GDPS",
        "GLUE_CONNECTION_NAME",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "RANDOMISE_PLAN_IDS"
    ]
)

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args["JOB_NAME"], args)

# -----------------------------
# Get database details from connection
# -----------------------------
connection_options = glue_context.extract_jdbc_conf(
    args["GLUE_CONNECTION_NAME"]
)

JDBC_URL = connection_options["fullUrl"]
DB_USER = connection_options["user"]
DB_PASSWORD = connection_options["password"]

# -----------------------------
# Read JSON (optimised parsing)
# -----------------------------
df_plans = (
    spark.read
        .option("multiLine", "true")
        .json(args["INPUT_S3_PATH_GDPlans"])
        .cache()
)

df_providers = (
    spark.read
        .option("multiLine", "true")
        .json(args["INPUT_S3_PATH_GD_P"])
        .cache()
)

df_charges = (
    spark.read
        .option("multiLine", "true")
        .json(args["INPUT_S3_PATH_GDPDC"])
        .cache()
)

df_measures = (
    spark.read
        .option("multiLine", "true")
        .json(args["INPUT_S3_PATH_GDPM"])
        .cache()
)

df_savings = (
    spark.read
        .option("multiLine", "true")
        .json(args["INPUT_S3_PATH_GDPS"])
        .cache()
)

print(f"Number of plan records read: {df_plans.count()}")
print(f"Number of providers records read: {df_providers.count()}")
print(f"Number of charges records read: {df_charges.count()}")
print(f"Number of measures records read: {df_measures.count()}")
print(f"Number of savings records read: {df_savings.count()}")

# -----------------------------
# Get charges, measures, and savings into right shape
# -----------------------------
df_charges_mod = (
    df_charges.filter(F.col("Active") == True)
    .groupBy("PlanId")
    .agg(
        F.sort_array(
            F.collect_list(
                F.struct(
                    F.date_format(F.to_timestamp("DailyChargeEndDate"), "yyyy-MM-dd HH:mm:ss.SSSSSS").alias("end_date"),
                    F.date_format(F.to_timestamp("DailyChargeStartDate"), "yyyy-MM-dd HH:mm:ss.SSSSSS").alias("start_date"),
                    F.col("DailyCharge").alias("daily_charge")
                )
        )
        ).alias("charges")
    ).select(
           F.col("PlanId"),
           F.to_json(F.col("charges")).alias("charges")
    )
)

df_measures_mod = (
    df_measures.groupBy("PlanId")
    .agg(
        F.sort_array(
            F.collect_list(
                F.struct(
                    F.col("Description").alias("product"),
                    F.date_format(F.to_timestamp("MeasureLastPayment"), "yyyy-MM-dd HH:mm:ss.SSSSSS").alias("repaid_date"),
                )
            )
        ).alias("measures")
    ).select(
           F.col("PlanId"),
           F.to_json(F.col("measures")).alias("measures")
    )
)

df_savings_mod = (
    df_savings.groupBy("PlanId")
    .agg(
        F.sort_array(
            F.collect_list(
                F.struct(
                    F.col("SavingsTotalYearly").alias("savings_total_yearly"),
                    F.col("SavingsGasYearly").alias("savings_gas_yearly"),
                    F.col("SavingsElectricityYearly").alias("savings_electricity_yearly"),
                    F.col("SavingsOtherYearly").alias("savings_other_yearly")
                )
            )
        ).alias("savings")
    ).select(
           F.col("PlanId"),
           F.to_json(F.col("savings")).alias("savings_scotland")
    )
)

# -----------------------------
# Get a list of the plan ids we want - first the most recent plan id from each series, then select any that still have a status of live
# -----------------------------
max_ids = (
    df_plans.groupBy("PlanId")
            .agg(F.max("Id").alias("Id"))
)

# the plan record has a status field (GreenDealPlanStatusId) that is updated with the plan update:
#   Id           Title
#   1            NEW
#   2            PENDING
#   3            CANCELLED
#   4            LIVE
#   5            COMPLETED

live_plans = (
    max_ids.alias("m")
        .join(
            df_plans.alias("p"),
            (F.col("m.PlanId") == F.col("p.PlanId")) &
            (F.col("m.Id") == F.col("p.Id")),
            "inner"
        )
        .filter(
            (F.col("PostInstall_RRN").isNotNull()) &
            (F.col("p.GreenDealPlanStatusId") == 4)
        )
        .select("p.*")
)

# -----------------------------
# Join plans to charges, measures, savings, and providers
# -----------------------------

result = (
    live_plans.alias("p")
    .join(df_charges_mod, ["PlanId"], "left")
    .join(df_savings_mod, ["PlanId"], "left")
    .join(df_measures_mod, ["PlanId"], "left")
    .join(
        df_providers.alias("pr"),
        F.concat(F.lit("GDP"), F.col("p.ProviderMPID"))
            == F.col("pr.MPID"),
        "left"
        )

    )

randomise_plan_ids = (
    args["RANDOMISE_PLAN_IDS"].lower() == "true"
)

if randomise_plan_ids:
    result = result.withColumn(
        "PlanId",
        F.concat(
            F.substring("PlanId", 1, 2),
            F.lpad((F.abs(F.xxhash64("PlanId")) % 10000000000).cast("string"), 10, "0")
            )
        )

# -----------------------------
# Build update dataset
# -----------------------------
plan_updates = (
    result.select(
        F.to_timestamp("CreatedDate").alias("start_date"),
        F.to_timestamp("EndDate").alias("end_date"),
        F.col("CompanyName").alias("provider_name"),
        F.col("GreenDealEnquireiesTel").alias("provider_telephone"),
        F.col("GreenDealEnquiriesEmail").alias("provider_email"),
        F.col("InterestRate").alias("interest_rate"),
        F.col("InterestRateIsFixed").alias("fixed_interest_rate"),
        F.to_timestamp("InterestRateUpliftDate").alias("charge_uplift_date"),
        F.col("IsCCA").alias("cca_regulated"),
        F.col("StructureChangedSinceAgreement").alias("structure_changed"),
        F.col("MeasuresRemovedFromProperty").alias("measures_removed"),
        F.coalesce(F.col("measures"), F.lit("[]")).alias("measures"),
        F.coalesce(F.col("charges"), F.lit("[]")).alias("charges"),
        F.lit("[]").alias("savings"),
        F.col("PlanId").alias("green_deal_plan_id"),
        F.coalesce(F.col("savings_scotland"), F.lit("[]")).alias("savings_scotland")
    )
        # important for large datasets → improves JDBC parallelism
    .distinct()
    .repartition(50)
    .cache()
)

assessment_id_updates = (
    result.select(
        F.col("PostInstall_RRN").alias("assessment_id"),
        F.col("PlanId").alias("green_deal_plan_id")
    )
    .filter(F.col("assessment_id").isNotNull())
    .distinct()
    .repartition(50)
    .cache()
)

plan_update_count = plan_updates.count()
assessment_id_update_count = assessment_id_updates.count()

print(f"Number of plan updates to process: {plan_update_count}")
print(f"Number of assessment id updates to process: {assessment_id_update_count}")

# -----------------------------
# Write staging table (overwrite)
# -----------------------------
(
    plan_updates.write
        .format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "public.green_deal_plans_updates")
        .option("user", DB_USER)
        .option("password", DB_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .option("batchsize", "10000")
        .mode("overwrite")
        .save()
)

(
    assessment_id_updates.write
        .format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", "scotland.green_deal_assessments_updates")
        .option("user", DB_USER)
        .option("password", DB_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .option("batchsize", "10000")
        .mode("overwrite")
        .save()
)

# -----------------------------
# Apply update in Postgres
# -----------------------------
conn = pg8000.connect(
    host=args["DB_HOST"],
    port=int(args["DB_PORT"]),
    database=args["DB_NAME"],
    user=DB_USER,
    password=DB_PASSWORD
)

cur = None

try:
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO public.green_deal_plans (
            start_date,
            end_date,
            provider_name,
            provider_telephone,
            provider_email,
            interest_rate,
            fixed_interest_rate,
            charge_uplift_date,
            cca_regulated,
            structure_changed,
            measures_removed,
            measures,
            charges,
            savings,
            green_deal_plan_id,
            savings_scotland
        )
        SELECT
            u.start_date,
            u.end_date,
            u.provider_name,
            u.provider_telephone,
            u.provider_email,
            u.interest_rate,
            u.fixed_interest_rate,
            u.charge_uplift_date,
            u.cca_regulated,
            u.structure_changed,
            u.measures_removed,
            u.measures::jsonb,
            u.charges::jsonb,
            u.savings::jsonb,
            u.green_deal_plan_id,
            u.savings_scotland::jsonb
        FROM public.green_deal_plans_updates u
        ON CONFLICT (green_deal_plan_id) DO NOTHING;
    """)

    plans_inserted = cur.rowcount

    cur.execute("""
        INSERT INTO scotland.green_deal_assessments (
            assessment_id,
            green_deal_plan_id
        )
        SELECT
            u.assessment_id,
            u.green_deal_plan_id
        FROM scotland.green_deal_assessments_updates u
        INNER JOIN scotland.assessments a
            ON a.assessment_id = u.assessment_id
        INNER JOIN public.green_deal_plans p
            ON p.green_deal_plan_id = u.green_deal_plan_id
        ON CONFLICT (green_deal_plan_id, assessment_id) DO NOTHING;
    """)


    assessments_inserted = cur.rowcount

    cur.execute("DROP TABLE IF EXISTS public.green_deal_plans_updates")
    cur.execute("DROP TABLE IF EXISTS scotland.green_deal_assessments_updates")

    conn.commit()

    print(f"Number of plans inserted: {plans_inserted}")
    print(f"Number of assessments inserted: {assessments_inserted}")

except Exception:

    conn.rollback()
    raise

finally:
    if cur is not None:
        cur.close()

    conn.close()

job.commit()