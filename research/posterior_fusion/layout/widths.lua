-- Pandoc's widthless LaTeX table columns cannot wrap. Estimate relative widths from
-- the longest cell in each column, with a floor for headings and diminishing returns.
function Table(tbl)
  local totals = {}
  local heads = {}
  if tbl.head and #tbl.head.rows > 0 then
    for i,cell in ipairs(tbl.head.rows[1].cells) do
      local h=pandoc.utils.stringify(cell.contents)
      heads[i]=h
      totals[i]=math.max(4,utf8.len(h)*1.25)
    end
  end
  for _,body in ipairs(tbl.bodies) do
    for _,row in ipairs(body.body) do
      for i,cell in ipairs(row.cells) do
        local s=pandoc.utils.stringify(cell.contents)
        local normal=s:gsub("[a-zA-Z0-9_]",".")
        totals[i]=math.max(totals[i] or 4, math.min(115,utf8.len(normal)))
      end
    end
  end
  local sum=0
  for i=1,#tbl.colspecs do
    totals[i]=math.max(3.1, math.min(14,math.sqrt(totals[i] or 9)))
    sum=sum+totals[i]
  end
  for i,spec in ipairs(tbl.colspecs) do spec[2]=totals[i]/sum end
  return tbl
end
