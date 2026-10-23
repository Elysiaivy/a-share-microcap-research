// All performance data is rendered in HTML; the switch only changes visibility.
document.querySelectorAll('[data-period]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-period]').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
      document.getElementById(item.dataset.period).hidden = !active;
    });
    document.querySelector('caption').textContent = '五档组合与中证1000绩效：' + button.textContent;
  });
});
