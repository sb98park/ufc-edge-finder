-- Octane Alpha — identity and entitlement schema
--
-- Applied to the Supabase project kmifrsmgypghjwmzoffd.
-- Run this in the SQL editor, or with `supabase db push` once the CLI is set up.
--
-- IDENTITY IS KEYED ON THE auth.users UUID, NEVER ON EMAIL. That is not an
-- arbitrary preference. When Sign in with Apple is added in Phase 2, an
-- existing Google user signing in with Apple must LINK to their account
-- rather than create a second one -- and Apple's Hide My Email returns a
-- private relay address that will not match the address already on file.
-- Any join on email silently breaks for exactly those users. profiles.id is
-- the UUID, stripe_customer_id hangs off it, and identity linking later is a
-- supported Supabase operation instead of a data migration.

-- ---------------------------------------------------------------------------
-- profiles: one row per account, created automatically on signup
-- ---------------------------------------------------------------------------

create table if not exists public.profiles (
  id                  uuid primary key references auth.users (id) on delete cascade,
  email               text        not null,
  stripe_customer_id  text        unique,

  -- TRIAL ELIGIBILITY LIVES HERE, NOT IN STRIPE. Stripe will happily grant a
  -- fresh trial to a new customer object, and a new customer object is one
  -- new email address away. This column is what makes "one trial per person"
  -- mean anything.
  trial_used_at       timestamptz,

  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now()
);

comment on column public.profiles.trial_used_at is
  'Set when a trial starts. Non-null blocks further trials for this account.';

-- ---------------------------------------------------------------------------
-- subscriptions: mirror of Stripe, written only by the webhook
-- ---------------------------------------------------------------------------

create table if not exists public.subscriptions (
  id                     bigint generated always as identity primary key,
  user_id                uuid        not null references public.profiles (id) on delete cascade,
  stripe_subscription_id text        not null unique,
  status                 text        not null,
  price_id               text        not null,
  plan_interval          text        not null check (plan_interval in ('month', 'year')),
  trial_end              timestamptz,
  current_period_end     timestamptz,
  cancel_at_period_end   boolean     not null default false,
  created_at             timestamptz not null default now(),
  updated_at             timestamptz not null default now()
);

-- `interval` is a reserved word in Postgres; plan_interval avoids quoting it
-- at every call site.

create index if not exists subscriptions_user_status_idx
  on public.subscriptions (user_id, status);

create index if not exists subscriptions_stripe_id_idx
  on public.subscriptions (stripe_subscription_id);

-- ---------------------------------------------------------------------------
-- entitlement
-- ---------------------------------------------------------------------------

-- PAST_DUE COUNTS AS ENTITLED, DELIBERATELY. Stripe is still retrying the
-- card at that point. Cutting access off mid-dunning over an expired card or
-- a bank blip is how a recoverable payment becomes a cancellation and a
-- support complaint. Access ends when Stripe gives up and the status moves to
-- canceled or unpaid.
create or replace function public.is_member(uid uuid)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists (
    select 1
      from public.subscriptions s
     where s.user_id = uid
       and s.status in ('trialing', 'active', 'past_due')
       and (s.current_period_end is null or s.current_period_end > now())
  );
$$;

comment on function public.is_member(uuid) is
  'True while the account should see member content. Mirrored into Cloudflare KV by the Stripe webhook so the edge never has to call this.';

-- ---------------------------------------------------------------------------
-- automatic profile creation
-- ---------------------------------------------------------------------------

-- Without this, a user exists in auth.users with no profile row, and the
-- first webhook trying to attach a Stripe customer fails on the foreign key.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, coalesce(new.email, ''))
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------------
-- updated_at maintenance
-- ---------------------------------------------------------------------------

create or replace function public.touch_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists profiles_touch on public.profiles;
create trigger profiles_touch before update on public.profiles
  for each row execute function public.touch_updated_at();

drop trigger if exists subscriptions_touch on public.subscriptions;
create trigger subscriptions_touch before update on public.subscriptions
  for each row execute function public.touch_updated_at();

-- ---------------------------------------------------------------------------
-- row level security
-- ---------------------------------------------------------------------------
--
-- THE ANON KEY IS PUBLIC BY DESIGN AND CARRIES NO PRIVILEGES OF ITS OWN.
-- Everything it can do is what these policies allow. A Supabase project with
-- the anon key published and RLS off is simply an open database, which is
-- why no table here is created without it.

alter table public.profiles      enable row level security;
alter table public.subscriptions enable row level security;

-- Read your own profile.
drop policy if exists profiles_select_own on public.profiles;
create policy profiles_select_own on public.profiles
  for select using (auth.uid() = id);

-- Update your own profile, but only fields a user has any business changing.
-- Note what is NOT grantable here: trial_used_at and stripe_customer_id are
-- never writable by the user. A client that could clear trial_used_at could
-- grant itself unlimited free trials.
drop policy if exists profiles_update_own on public.profiles;
create policy profiles_update_own on public.profiles
  for update using (auth.uid() = id) with check (auth.uid() = id);

revoke update on public.profiles from anon, authenticated;
grant  update (email) on public.profiles to authenticated;

-- Read your own subscriptions. There is no insert/update/delete policy at
-- all, for anyone: subscription rows are written exclusively by the Stripe
-- webhook using the service role, which bypasses RLS. A user being able to
-- write their own subscription row is a user being able to grant themselves
-- a membership.
drop policy if exists subscriptions_select_own on public.subscriptions;
create policy subscriptions_select_own on public.subscriptions
  for select using (auth.uid() = user_id);
