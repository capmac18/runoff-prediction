import camelot

file = "Ozarkhed_Rainfall_Runoff.pdf"

tables = camelot.read_pdf(file)

df = tables[0].df

df=df[1:]
data_list = df.astype(float).values.tolist()
print(df)

year= df[0].astype(int).values.tolist()
X = np.array(df[1])
print(year)

