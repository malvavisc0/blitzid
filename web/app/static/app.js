document.querySelectorAll("[data-tabs]").forEach((tabList) => {
  const tabs = [...tabList.querySelectorAll('[role="tab"]')];

  function selectTab(active) {
    for (const tab of tabs) {
      const selected = tab === active;
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (panel) panel.hidden = !selected;
    }
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else return;

      event.preventDefault();
      selectTab(tabs[next]);
      tabs[next].focus();
    });
  });

  tabList.hidden = false;
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  const container = document.getElementById(button.dataset.copy);
  if (!container) return;

  const original = button.textContent;
  let reset;

  button.addEventListener("click", async () => {
    // A container of language panes copies the one on screen.
    const target = container.matches("pre") ? container : container.querySelector("pre:not([hidden])");
    if (!target) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(target.textContent);
      } else {
        const range = document.createRange();
        range.selectNodeContents(target);
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
        const copied = document.execCommand("copy");
        selection?.removeAllRanges();
        if (!copied) throw new Error("Clipboard unavailable");
      }
      button.textContent = "Copied";
    } catch {
      button.textContent = "Failed";
    }

    window.clearTimeout(reset);
    reset = window.setTimeout(() => {
      button.textContent = original;
    }, 1500);
  });

  button.hidden = false;
});

const header = document.querySelector(".site-header");
if (header) {
  const markScrolled = () => header.toggleAttribute("data-scrolled", window.scrollY > 0);
  markScrolled();
  window.addEventListener("scroll", markScrolled, { passive: true });
}

const year = document.querySelector("[data-year]");
if (year) year.textContent = String(new Date().getFullYear());
