
module wheel(dia=30, tyre=7.5, width=15) {
	union() {
		translate([0,-width/2,0]) {
			rotate([-90,0,0]) {
//gt label: ***
color([0,0,0])
				difference() {
//gt label: ***
color([0,0,0])
					cylinder(h=width, r1=dia, r2=dia);
					translate([0,0,-0.5])
//gt label: ***
color([0,0,0])
					cylinder(h=width + 1, r1=dia - tyre, r2=dia - tyre);
				}
//gt label: ***
color([0,0,0])
				cylinder(h=width, r1=dia/6, r2=dia/6);
				translate([0,0,-width/2])
//gt label: ***
color([0,0,0])
				cylinder(h=width * 2, r1=dia/10, r2=dia/9);
			}
		

			translate([0,width/2,0])
			spokes(length=dia - 1, count=5, dia=width/2);
		}
	}
}

module spokes(length=30, count=10, dia=2.5) {
	angle = 360 / count;
	for (i = [0 : count]) {
		rotate([angle * i,0,90])
//gt label: ***
color([0,0,0])
		cylinder(h=length, r1=dia/4, r2=dia);
	}
}

module frame(length, dia=5) {
	translate([0,0,50])
	rotate([0,90,0])
//gt label: ***
color([0,0,0])
	cylinder(h=length * 0.5, r1=dia, r2=dia);



	


//gt label: ***
color([0,0,0])
	cylinder(h=length * 0.6, r1=dia*1.5, r2=dia*1.5);


	translate([1 - length/2, 1-dia , 0])
	rotate([0,90,-5])
//gt label: ***
color([0,0,0])
	cylinder(length * 0.5, r1=dia/2, r2=dia);
	translate([1 - length/2, dia , 0])
	rotate([0,90,5])
//gt label: ***
color([0,0,0])
	cylinder(length * 0.5, r1=dia/2, r2=dia);


	translate([1 - length/2, 1-dia , 0])
	rotate([0,45,-2.5])
//gt label: ***
color([0,0,0])
	cylinder(length * 0.7, r1=dia/2, r2=dia);
	translate([1 - length/2, dia , 0])
	rotate([0,45,2.5])
//gt label: ***
color([0,0,0])
	cylinder(length * 0.7, r1=dia/2, r2=dia);


	translate([length * 0.5 ,dia * 1.2,0])
//gt label: ***
color([0,0,0])
	cylinder(h=length * 0.4, r1=dia/2, r2=dia);
	translate([length * 0.5,-dia * 1.2,0])	
//gt label: ***
color([0,0,0])
	cylinder(h=length * 0.4, r1=dia/2, r2=dia);


	translate([length * 0.5,0,length * 0.3])	
//gt label: ***
color([0,0,0])
	cylinder(h=length * 0.3, r1=dia*1.5, r2=dia*1.5);


	rotate([0,45,0])
//gt label: ***
color([0,0,0])
	cylinder(h=length * 0.7, r1=dia, r2=dia);


	translate([length * 0.5, (length * 0.75)/2, length * 0.8])
	handlebars(length * 0.75, dia=4);


	translate([0,0,length*0.7])
	seat(length * 0.3, dia=4);


	translate([0,-10,0])
	crank(length * 0.15, length * 0.25);
}

module crank(length, width) {
	rotate([90,0,0]) {	
//gt label: ***
color([0,0,0])
		cylinder(h=width/5, r1=length, r2=length);
		translate([0,0,-width])	
//gt label: ***
color([0,0,0])
		cylinder(h=width, r1=length/2, r2=length/2);
	}
	

	translate([length * 1.75/2,width,0])
//gt label: ***
color([0,0,0])
	cube([length * 1.75, length/4, length/2], center=true);

	translate([-(length * 1.75/2),-(width/5),0])
//gt label: ***
color([0,0,0])
	cube([length * 1.75, length/4, length/2], center=true);


}

module seat(length, dia=2.5) {
	rotate([0,90,0])
//gt label: ***
color([0,0,0])
	difference() {
		translate([0,0,-length/2])
//gt label: ***
color([0,0,0])
		cylinder(h=length, r1=length/2, r2=length/3);

		translate([length/2 -0.5, 0,0])
//gt label: ***
color([0,0,0])
		cube(length + 1, center=true);	
	}
	translate([0,0,1-length/2 ])
//gt label: ***
color([0,0,0])
	cylinder(h=length/2, r1=dia, r2=dia);
}

module handlebars(length, dia=2.5) {

	translate([0,-length/2,-length/3])
//gt label: ***
color([0,0,0])
	cylinder(h=length/3, r1=dia, r2=dia);
	

	rotate([90,0,0])
//gt label: ***
color([0,0,0])
	cylinder(h=length, r1=dia, r2=dia);
	

	translate([dia/2 - 1.5,-dia,0])
	rotate([0,-90,0])
//gt label: ***
color([0,0,0])
	cylinder(h=length*0.3, r1=dia,r2=dia/2);

	translate([dia/2 - 1.5,-length + dia,0])
	rotate([0,-90,0])
//gt label: ***
color([0,0,0])
	cylinder(h=length*0.3, r1=dia,r2=dia/2);
}


union() {
	frame(length=100);

	translate([-50,0,0])
	wheel();

	translate([50,0,0])
	wheel();
}