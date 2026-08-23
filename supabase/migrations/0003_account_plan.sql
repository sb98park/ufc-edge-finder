-- What to CALL someone's access, as distinct from whether they have it.
--
-- is_member() stays the only thing that decides what a person can see. This
-- function decides only what the account panel prints, and the difference
-- matters: a bug here shows the wrong word, a bug there gives away the
-- product. Nothing downstream is allowed to gate on the string this returns.
--
-- Three levels, matching how the product is sold:
--
--   basic     signed in, no subscription -- every card already graded
--   pro       a live subscription (trialing, active, or past_due)
--   lifetime  a permanent grant, set by hand via profiles.lifetime_access
--
-- Lifetime is checked FIRST and deliberately outranks a subscription: an
-- account that has both should read as the thing that does not expire.

create or replace function public.account_plan(uid uuid)
returns text
language sql
stable
security definer
set search_path = public
as $$
  select case
    when coalesce((select p.lifetime_access from public.profiles p where p.id = uid), false)
      then 'lifetime'
    when exists (
      select 1
        from public.subscriptions s
       where s.user_id = uid
         and s.status in ('trialing', 'active', 'past_due')
         and (s.current_period_end is null or s.current_period_end > now())
    ) then 'pro'
    else 'basic'
  end;
$$;

comment on function public.account_plan(uuid) is
  'Display label for an account''s access level: basic | pro | lifetime. '
  'NOT an entitlement check -- use is_member(uuid) for that. Kept in step '
  'with is_member: anything it calls pro or lifetime, is_member must call true.';

-- The two functions must never disagree about who has access. This is the
-- assertion, run at migration time against real rows rather than assumed:
-- every account account_plan() calls paid, is_member() must also call true,
-- and every account it calls basic, is_member() must call false.
do $$
declare
  mismatches integer;
begin
  select count(*) into mismatches
    from public.profiles p
   where (public.account_plan(p.id) in ('pro', 'lifetime')) is distinct from public.is_member(p.id);

  if mismatches > 0 then
    raise exception
      'account_plan and is_member disagree on % account(s) -- fix before shipping, '
      'a label that outruns entitlement is how a free account gets told it is paid',
      mismatches;
  end if;
end $$;

grant execute on function public.account_plan(uuid) to authenticated, service_role;
