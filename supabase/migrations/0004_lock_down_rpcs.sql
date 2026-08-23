-- Stop anyone with the public anon key from asking about anyone's account.
--
-- Postgres grants EXECUTE on new functions to PUBLIC by default, and the
-- `grant execute ... to authenticated, service_role` in 0003 is additive --
-- it does not take that default away. Both functions are SECURITY DEFINER,
-- so the default made them answerable by the anon key, which is embedded in
-- the page source and belongs to anyone who views it:
--
--   $ curl -s .../rest/v1/rpc/account_plan -H "apikey: <anon>" \
--          -d '{"uid":"<someone>"}'
--   "basic"
--
-- Given a user's id, an anonymous caller could read whether that person is
-- paying. Ids are uuids and are not published anywhere, so this was never
-- open enumeration -- but "you also need a uuid" is not an access control,
-- and neither function has any business answering an anonymous caller.
--
-- Nothing legitimate is affected. Only the Cloudflare Worker calls these, and
-- it authenticates with the service role, which is granted explicitly below.
-- No client-side code calls them at all: the browser asks the Worker's
-- /auth/whoami, and the Worker asks Postgres.
--
-- is_member() is included even though it predates account_plan() -- it has
-- the same exposure for the same reason, and fixing one while leaving the
-- other would just be tidier, not safer.

revoke execute on function public.is_member(uuid) from public, anon;
revoke execute on function public.account_plan(uuid) from public, anon;

grant execute on function public.is_member(uuid) to service_role;
grant execute on function public.account_plan(uuid) to service_role;

-- `authenticated` keeps is_member so a signed-in session can ask about
-- ITSELF through RLS-protected paths, which is how the policies in 0001 are
-- written. account_plan is a display label the Worker resolves and hands
-- back, so no client role needs it.
grant execute on function public.is_member(uuid) to authenticated;

-- Verify from the database's own catalog rather than trusting the statements
-- above to have matched the right signatures.
do $$
declare
  leaky text;
begin
  select string_agg(p.proname, ', ')
    into leaky
    from pg_proc p
    join pg_namespace n on n.oid = p.pronamespace
   where n.nspname = 'public'
     and p.proname in ('is_member', 'account_plan')
     and (has_function_privilege('anon', p.oid, 'execute')
          or has_function_privilege('public', p.oid, 'execute'));

  if leaky is not null then
    raise exception 'still executable by anon/public: % -- the revoke did not take', leaky;
  end if;

  raise notice 'ok: is_member and account_plan are service_role/authenticated only';
end $$;
