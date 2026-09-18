-- Run this once in the Supabase SQL editor (Project -> SQL Editor -> New query)
-- for the project you'll point SUPABASE_URL / SUPABASE_KEY at.

create table if not exists transactions (
    id bigint generated always as identity primary key,
    ingested_at timestamptz not null default now(),
    original_time numeric not null,       -- the dataset's Time column (seconds since first transaction)
    amount numeric not null,
    anomaly_score double precision not null,
    threshold double precision not null,
    flagged boolean not null,
    is_synthetic boolean not null default false,
    true_class integer,                   -- demo-only: the real Class label, shown in the dashboard
                                           -- for reference AFTER scoring. The model itself never sees
                                           -- this at inference time, same as in Phase 1-3.
    model_name text not null default 'isolation_forest'
);

create index if not exists transactions_ingested_at_idx on transactions (ingested_at desc);
create index if not exists transactions_flagged_idx on transactions (flagged);

-- Row Level Security: the backend writes using the service_role key,
-- which bypasses RLS entirely. The dashboard reads using the anon key,
-- so it needs an explicit read policy below.
alter table transactions enable row level security;

create policy "Public read access" on transactions
    for select
    using (true);
