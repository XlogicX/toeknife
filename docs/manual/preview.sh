#!/bin/sh
# Render the manual to HTML for reading locally, links and screenshots working.
#
# The pages link to each other as .md, which is right on GitHub. Locally that
# opens raw markdown, so each page is rendered with a filter that points those
# links at the .html it writes instead. Screenshots are embedded.
#
#   docs/manual/preview.sh            # renders, then opens the index
#   docs/manual/preview.sh /some/dir  # renders there instead
#
# The default is docs/manual/preview-html/ (gitignored), not /tmp: a snap
# browser cannot read /tmp, and when handed one file from there it sees only
# that file, so every link between pages is dead.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-$here/preview-html}
mkdir -p "$out"
out=$(cd "$out" && pwd)
filter="$out/md-links.lua"
cat > "$filter" <<'LUA'
function Link(el)
  el.target = el.target:gsub("%.md$", ".html"):gsub("%.md#", ".html#")
  return el
end
LUA
cd "$here"
for page in *.md; do
  title=$(sed -n 's/^# //p' "$page" | head -1)
  pandoc "$page" -f gfm -s --embed-resources --lua-filter "$filter" \
    --metadata title="${title:-$page}" -o "$out/${page%.md}.html"
done
rm -f "$filter"
echo "rendered $(ls "$out"/*.html | wc -l) pages into $out"
[ -n "$1" ] || xdg-open "$out/README.html" >/dev/null 2>&1 || true
