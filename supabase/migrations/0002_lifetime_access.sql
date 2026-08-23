-- Lifetime / complimentary access, granted outside Stripe.
--
-- WHY THIS IS NOT A FAKE SUBSCRIPTION ROW. The quick way to give the owner
-- permanent access is to insert a subscriptions row with status 'active' and
-- current_period_end set to 2099. Three things go wrong with that:
--
--   1. The Stripe webhook is the only writer of that table and reconciles it
--      against Stripe. A row Stripe has never heard of is a row a future sync
--      can legitimately delete or overwrite -- silently, at the worst moment.
--   2. It corrupts every revenue figure. A $0 "active" subscription counts as
--      a subscriber in MRR, churn and conversion, and you would be debugging
--      that discrepancy months later.
--   3. It hides intent. "This person pays nothing and always has access" is a
--      fact worth being able to see, not something to infer from a suspicious
--      period end date.
--
-- An explicit column says what is true, survives any Stripe reconciliation,
-- and is trivially auditable: `select email from profiles where
-- lifetime_access`.

alter table public.profiles
  add column if not exists lifetime_access boolean not null default false,
  add column if not exists access_note     text;

comment on column public.profiles.lifetime_access is
  'Permanent entitlement granted outside Stripe: owner, comped press, lifetime backers. Never set by user-facing code.';
comment on column public.profiles.access_note is
  'Why the grant exists, so a future reader does not have to guess.';

-- Entitlement now has two independent sources. Order matters only for
-- readability -- a lifetime grant short-circuits before touching the
-- subscriptions table at all.
create or replace function public.is_member(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select
    coalesce((select p.lifetime_access from public.profiles p where p.id = uid), false)
    or exists (
      select 1
        from public.subscriptions s
       where s.user_id = uid
         and s.status in ('trialing', 'active', 'past_due')
         and (s.current_period_end is null or s.current_period_end > now())
    );
$$;

-- The column is NOT user-writable. profiles already has update revoked with
-- only `email` granted back, so this inherits that protection -- but it is
-- restated here because the consequence of getting it wrong is every user
-- granting themselves a free lifetime membership.
revoke update on public.profiles from anon, authenticated;
grant  update (email) on public.profiles to authenticated;

-- ---------------------------------------------------------------------------
-- Backfill profiles for users who predate the trigger
-- ---------------------------------------------------------------------------
-- Postgres triggers do not apply retroactively. Any auth.users row created
-- BEFORE migration 0001 installed on_auth_user_created has no profile, and
-- every lookup keyed on profiles silently misses it -- an update matching
-- zero rows looks exactly like an update that worked.
--
-- This bit us for real: the accounts created while testing SMTP predated the
-- schema, so the owner's own lifetime grant matched nothing on the first
-- attempt. Safe to re-run; it only ever fills gaps.

insert into public.profiles (id, email)
select id, coalesce(email, '')
  from auth.users
    on conflict (id) do nothing;

-- ---------------------------------------------------------------------------
-- Granting the owner account
-- ---------------------------------------------------------------------------
-- Run this ONCE, after signing in at least once so the auth user exists.
-- Adjust the address if the owner account uses a different one.
--
--   update public.profiles
--      set lifetime_access = true,
--          access_note     = 'Owner account'
--    where email = 'octanealphainfo@gmail.com';
--
-- Verify:
--
--   select email, lifetime_access, access_note from public.profiles
--    where lifetime_access;
--
--   select public.is_member(id) from public.profiles
--    where email = 'octanealphainfo@gmail.com';
