import { useEffect, useRef, useState } from "react";
import { User } from "./api";

type Props = {
  user: User | null;
  onLogout: () => void;
};

export default function AccountMenu({ user, onLogout }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const initial = (user?.name || user?.email || "?").trim().charAt(0).toUpperCase();

  return (
    <div className="account-menu" ref={ref}>
      <button
        className="account-btn"
        type="button"
        aria-label="Account menu"
        onClick={() => setOpen((v) => !v)}
      >
        {user?.avatar_url ? (
          <img src={user.avatar_url} alt="" className="account-avatar" referrerPolicy="no-referrer" />
        ) : (
          <span className="account-initial">{initial}</span>
        )}
      </button>
      {open && (
        <div className="account-dropdown">
          <div className="account-meta">
            <strong>{user?.name || "Account"}</strong>
            <span>{user?.email}</span>
          </div>
          <button
            className="btn"
            type="button"
            onClick={() => {
              setOpen(false);
              onLogout();
            }}
          >
            Log out
          </button>
        </div>
      )}
    </div>
  );
}
